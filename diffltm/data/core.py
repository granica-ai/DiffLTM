"""Versioned tables and one shared run-seeded 70/10/20 partition."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..artifacts import atomic_write, file_identity, identity, write_json

TABLE_VERSION = "v1"


def dataset_identity(manifest):
    """Data compatibility excludes provenance, paths and cosmetic version labels."""
    contents = "content_hashes" if "content_hashes" in manifest else "files"
    return identity({k: manifest[k] for k in ("name", "schema", contents)})


def verify_dataset(saved, current, frame=None):
    same_schema = all(saved[k] == current[k] for k in ("name", "schema"))
    if same_schema:
        if "content_hashes" in saved and "content_hashes" in current:
            if saved["content_hashes"] == current["content_hashes"]:
                return
        elif saved.get("files") == current.get("files"):
            return
        elif frame is not None:
            # The original release recorded a row-content hash inside a provenance-
            # dependent identity. Reconstruct it for old runs after a Parquet rewrite.
            # Reference frames were not included, so retain their original byte check.
            saved_refs = {k: v for k, v in saved["files"].items() if k != "table.parquet"}
            current_refs = {k: v for k, v in current["files"].items() if k != "table.parquet"}
            if saved_refs == current_refs:
                fields = {"name", "schema", "source", "version", "rows", "identity", "files"}
                legacy = dict(
                    schema=saved["schema"],
                    source=saved["source"],
                    version="corrected-v1",
                    rows=pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes().hex(),
                    extra={k: v for k, v in saved.items() if k not in fields},
                )
                if saved["identity"] == identity(legacy):
                    return
    raise ValueError("Dataset content/schema mismatch")


def frame_identity(frame):
    """Hash ordered values, column names and row IDs, independently of the writer."""
    rows = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="<u8").tobytes()
    return identity(dict(columns=list(frame.columns), rows=hashlib.sha256(rows).hexdigest()))


def table_contents(frame, reference=None):
    result = {"table.parquet": frame_identity(frame)}
    if reference is not None:
        result["reference.parquet"] = frame_identity(reference)
    return result


def split_identity(splits):
    """Hash ordered assignments independently of NPZ serialization."""
    digest = hashlib.sha256()
    for name in ("train", "validation", "test"):
        values = np.asarray(splits[name], dtype="<i8")
        digest.update(name.encode())
        digest.update(len(values).to_bytes(8, "little"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def validate_splits(splits, rows):
    combined = np.concatenate([splits[k] for k in ("train", "validation", "test")])
    if combined.dtype.kind not in "iu" or not np.array_equal(np.sort(combined), np.arange(rows)):
        raise ValueError("Invalid saved split")


def load_run_splits(run, table, checkpoint_splits=None):
    run = Path(run)
    saved = json.loads((run / "data.json").read_text())
    verify_dataset(saved["table"], table.manifest, table.frame)
    if file_identity(run / "split.npz") != saved["splits_sha256"]:
        raise ValueError("Saved split checksum mismatch")
    with np.load(run / "split.npz", allow_pickle=False) as archive:
        splits = {k: archive[k] for k in ("train", "validation", "test")}
    validate_splits(splits, len(table.frame))
    if checkpoint_splits is not None and any(
        not np.array_equal(splits[k], checkpoint_splits[k]) for k in splits
    ):
        raise ValueError("Checkpoint/saved split mismatch")
    return splits


@dataclass(frozen=True)
class Schema:
    numerical: list[str]
    categorical: list[str]
    target: str
    task: str
    integer: list[str] = field(default_factory=list)

    @property
    def columns(self):
        return self.numerical + self.categorical

    def __post_init__(self):
        if len(set(self.columns)) != len(self.columns) or self.target not in self.columns:
            raise ValueError("Duplicate columns or missing target")
        if self.task not in {"regression", "binclass", "multiclass"}:
            raise ValueError(self.task)
        if self.target not in (self.numerical if self.task == "regression" else self.categorical):
            raise ValueError("Target role disagrees with task")
        if not set(self.integer) <= set(self.numerical):
            raise ValueError("Integer decode roles must be numerical")


@dataclass
class Table:
    frame: pd.DataFrame
    schema: Schema
    manifest: dict
    reference: pd.DataFrame | None = None

    def __post_init__(self):
        if self.frame.index.has_duplicates or self.frame.index.name != "row_id":
            raise ValueError("Tables require unique stable row_id index")
        if list(self.frame.columns) != self.schema.columns:
            raise ValueError("Modeled columns disagree with schema")


def category_values(values):
    """Preserve observed zero codes; missing is a distinct ordinary level."""
    if pd.api.types.is_numeric_dtype(values.dtype):
        return values.map(
            lambda x: "__NA__" if pd.isna(x) else str(int(x)) if float(x).is_integer() else str(x)
        )
    return values.astype("string").fillna("__NA__").astype(str)


def make_table(frame, schema, name, source, reference=None, **extra):
    frame = frame.loc[:, schema.columns].copy()
    for col in schema.categorical:
        frame[col] = category_values(frame[col])
    frame.index = frame.index.astype(str)
    frame.index.name = "row_id"
    table = Table(
        frame,
        schema,
        dict(
            name=name,
            version=TABLE_VERSION,
            source=source,
            content_hashes=table_contents(frame, reference),
            rows=len(frame),
            schema=asdict(schema),
            **extra,
        ),
        reference,
    )
    table.manifest["identity"] = dataset_identity(table.manifest)
    return table


def load_table(name, cache="data"):
    root = Path(cache) / "tables" / TABLE_VERSION / name
    legacy = Path(cache) / "tables" / "corrected-v1" / name
    if not (root / "manifest.json").exists() and (legacy / "manifest.json").exists():
        root = legacy
    meta = root / "manifest.json"
    if meta.exists():
        manifest = json.loads(meta.read_text())
        if manifest.get("name") != name or "table.parquet" not in manifest.get("files", {}):
            raise ValueError("Table cache requires dataset name and file checksums")
        for filename, digest in manifest["files"].items():
            if file_identity(root / filename) != digest:
                raise ValueError(f"Cache content mismatch: {filename}")
        frame = pd.read_parquet(root / "table.parquet")
        reference = (
            pd.read_parquet(root / "reference.parquet")
            if "reference.parquet" in manifest["files"]
            else None
        )
        contents = table_contents(frame, reference)
        if "content_hashes" in manifest and manifest["content_hashes"] != contents:
            raise ValueError("Cache content hash mismatch")
        manifest = dict(manifest, version=TABLE_VERSION, content_hashes=contents)
        manifest["identity"] = dataset_identity(manifest)
        return Table(frame, Schema(**manifest["schema"]), manifest, reference)
    if name.startswith("sc_reparam_"):
        from .synthetic import DEFAULT_CONFIG, generate_panel

        panel = generate_panel()
        retained = panel.loc[~panel.TREATED_POST].copy()
        maximum = float(retained.Y.max())
        variant = name.removeprefix("sc_reparam_")
        target = {"level": "Y", "log": "LOG_Y", "quadratic": "QUAD_Y"}[variant]
        retained[target] = {
            "level": lambda y: y,
            "log": np.log,
            "quadratic": lambda y: y * (maximum - y),
        }[variant](retained.Y)
        schema = Schema(["U", target], ["S", "T"], target, "regression")
        table = make_table(
            retained,
            schema,
            name,
            "pinned-dgp-42",
            panel,
            M=maximum,
            dgp=asdict(DEFAULT_CONFIG),
            exclusions="S=6,T in [8,9,10]",
            split_family="sc_reparam",
        )
    elif name.startswith("acs"):
        from .acs import build

        table = build(name, cache)
    elif name.startswith("cps_wage_"):
        from .cps import build

        table = build(name, cache)
    else:
        from .openml import build

        table = build(name, cache)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write(root / "table.parquet", lambda p: table.frame.to_parquet(p))
    files = ["table.parquet"]
    if table.reference is not None:
        atomic_write(root / "reference.parquet", lambda p: table.reference.to_parquet(p))
        files.append("reference.parquet")
    table.manifest["files"] = {f: file_identity(root / f) for f in files}
    table.manifest["identity"] = dataset_identity(table.manifest)
    write_json(meta, table.manifest)
    return table


def split_table(table, seed, cache="data"):
    key = identity(
        [table.manifest["identity"], table.manifest.get("exclusions"), "70/10/20-v1", seed]
    )
    path = Path(cache) / "splits" / f"{key}.npz"
    source = path
    # Reuse the exact old assignments when an earlier table cache is present.
    legacy = Path(cache) / "tables" / "corrected-v1" / table.manifest["name"] / "manifest.json"
    if not source.exists() and legacy.exists():
        saved = json.loads(legacy.read_text())
        old_key = identity([saved["identity"], saved.get("exclusions"), "70/10/20-v1", seed])
        old_path = Path(cache) / "splits" / f"{old_key}.npz"
        if old_path.exists():
            try:
                verify_dataset(saved, table.manifest, table.frame)
            except ValueError:
                pass  # A different rebuilt table needs its own fresh partition.
            else:
                source = old_path
    if source.exists():
        if source == path:
            metadata = json.loads(path.with_suffix(".json").read_text())
            if file_identity(path) != metadata["sha256"]:
                raise ValueError("Cached split checksum mismatch")
        with np.load(source, allow_pickle=False) as z:
            result = {k: z[k] for k in ("train", "validation", "test")}
    else:
        ids = np.arange(len(table.frame))
        stratify = (
            table.frame[table.schema.target].to_numpy()
            if table.schema.task != "regression"
            else None
        )
        train_val, test = train_test_split(ids, test_size=0.2, random_state=seed, stratify=stratify)
        train, val = train_test_split(
            train_val,
            test_size=0.125,
            random_state=seed,
            stratify=None if stratify is None else stratify[train_val],
        )
        result = dict(train=train, validation=val, test=test)
        atomic_write(path, lambda p: _save_npz(p, result))
        write_json(path.with_suffix(".json"), dict(sha256=file_identity(path)))
    validate_splits(result, len(table.frame))
    return result, split_identity(result)


def _save_npz(path, values):
    with path.open("wb") as f:
        np.savez_compressed(f, **values)
