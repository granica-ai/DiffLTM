"""Old training artifacts remain usable while data changes are rejected."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import config

from diffltm import evaluation, trainer
from diffltm.artifacts import file_identity, identity, write_json
from diffltm.data.core import (
    dataset_identity,
    load_table,
    make_table,
    split_identity,
    split_table,
    verify_dataset,
)


def legacy_cache(tmp_path, table):
    cache = tmp_path / "data"
    root = cache / "tables/corrected-v1/fixture"
    root.mkdir(parents=True)
    table.frame.to_parquet(root / "table.parquet")
    manifest = dict(table.manifest, version="corrected-v1")
    manifest.pop("content_hashes")
    manifest["identity"] = identity(
        dict(
            schema=manifest["schema"],
            source=manifest["source"],
            version="corrected-v1",
            rows=pd.util.hash_pandas_object(table.frame, index=True).to_numpy().tobytes().hex(),
            extra={},
        )
    )
    manifest["files"] = {"table.parquet": file_identity(root / "table.parquet")}
    write_json(root / "manifest.json", manifest)
    return cache, root, manifest


def test_version_alias_preserves_bytes_and_split_membership(tmp_path, small_table):
    cache, root, old = legacy_cache(tmp_path, small_table)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    current = load_table("fixture", cache)
    assert current.manifest["version"] == "v1"
    assert current.manifest["identity"] == small_table.manifest["identity"]
    verify_dataset(old, current.manifest)
    splits, digest = split_table(current, 500, cache)
    assert digest == split_identity(splits)
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    assert not (cache / "tables/v1").exists()
    # Acquisition metadata and implementation labels cannot invalidate data.
    old.update(source={"reader": "older-reader", "implementation": "unused"})
    write_json(root / "manifest.json", old)
    assert load_table("fixture", cache).manifest["identity"] == current.manifest["identity"]
    saved_split = next((cache / "splits").glob("*.npz"))
    saved_split.write_bytes(saved_split.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Cached split checksum"):
        split_table(current, 500, cache)
    (root / "table.parquet").write_bytes(before["table.parquet"] + b"changed")
    with pytest.raises(ValueError, match="content mismatch"):
        load_table("fixture", cache)


def test_table_identity_ignores_parquet_encoding_and_checks_values(tmp_path, small_table):
    cache, root, legacy = legacy_cache(tmp_path, small_table)
    first = load_table("fixture", cache)
    new_root = cache / "tables/v1/fixture"
    new_root.mkdir(parents=True)
    first.frame.to_parquet(new_root / "table.parquet", compression=None, row_group_size=17)
    manifest = dict(
        first.manifest, files={"table.parquet": file_identity(new_root / "table.parquet")}
    )
    assert manifest["files"] != legacy["files"]
    write_json(new_root / "manifest.json", manifest)
    second = load_table("fixture", cache)
    assert first.frame.equals(second.frame)
    assert first.manifest["identity"] == second.manifest["identity"]
    verify_dataset(first.manifest, second.manifest)
    verify_dataset(legacy, second.manifest, second.frame)
    a, aid = split_table(first, 500, cache)
    b, bid = split_table(second, 500, cache)
    assert aid == bid and all(np.array_equal(a[k], b[k]) for k in a)
    # Even after updating the byte checksum, changed data must fail content checks.
    changed = second.frame.copy()
    changed.iloc[0, 0] += 1
    changed.to_parquet(new_root / "table.parquet")
    manifest["files"]["table.parquet"] = file_identity(new_root / "table.parquet")
    write_json(new_root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_table("fixture", cache)
    altered = make_table(changed, second.schema, "fixture", "different-source")
    altered.manifest["files"] = manifest["files"]
    with pytest.raises(ValueError, match="Dataset content/schema"):
        verify_dataset(legacy, altered.manifest, altered.frame)
    old_key = identity([legacy["identity"], None, "70/10/20-v1", 500])
    np.savez_compressed(cache / "splits" / f"{old_key}.npz", **a)
    # A stale legacy cache must not block splitting a deliberately rebuilt table.
    split_table(altered, 500, cache)


def test_table_identity_locks_reference_rows_and_order(small_table):
    table = small_table

    def build(frame, reference=None):
        return make_table(frame, table.schema, "fixture", "source", reference).manifest

    original = build(table.frame)
    assert dataset_identity(original) != dataset_identity(build(table.frame.iloc[::-1]))
    reference = table.frame.copy()
    with_reference = build(table.frame, reference)
    reference.iloc[0, 0] += 1
    assert dataset_identity(with_reference) != dataset_identity(build(table.frame, reference))


def test_plot_combines_legacy_and_content_identities(tmp_path, small_table, monkeypatch):
    import sys

    from scripts import plot_results

    cache, _, old = legacy_cache(tmp_path, small_table)
    current = load_table("fixture", cache).manifest
    paths = []
    for i, table in enumerate((old, current)):
        path = tmp_path / str(i) / "scores.csv"
        path.parent.mkdir()
        pd.DataFrame(
            [
                dict(
                    dataset="fixture",
                    dataset_identity=table["identity"],
                    method="unconditional",
                    training_seed=500,
                )
            ]
        ).to_csv(path, index=False)
        write_json(
            path.with_name("manifest.json"),
            dict(
                status="complete",
                table_manifest=table,
                files={"scores.csv": file_identity(path)},
                training_runs=[
                    dict(status="complete", run_id="old-run", run_dir=str(tmp_path / "missing"))
                ],
            ),
        )
        paths.append(str(path))
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(
        [
            dict(
                results=";".join(paths),
                output="comparison",
                seeds="[500]",
                plotting_function="format_results_table",
            )
        ]
    ).to_csv(inventory, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_results", "--manifest", str(inventory), "--output", str(tmp_path / "figures")],
    )
    monkeypatch.setattr(plot_results.plotting, "format_results_table", lambda frame: frame)
    plot_results.main()
    combined = pd.read_csv(tmp_path / "figures/comparison.csv")
    assert combined.dataset_identity.tolist() == [current["identity"]] * 2


def test_old_checkpoint_extra_hashes_and_fresh_evaluations(tmp_path, small_table, model_config):
    cache, _, legacy = legacy_cache(tmp_path, small_table)
    cfg = config(tmp_path, model_config)
    cfg["cache"] = str(cache)
    trainer.train(cfg)
    run = Path(cfg["output_dir"])
    checkpoint = torch.load(run / "selected.pt", map_location="cpu", weights_only=False)
    checkpoint["table_manifest"] = legacy
    checkpoint["implementation"] = {"trainer.py": "old-unused-hash"}
    torch.save(checkpoint, run / "selected.pt")
    status = json.loads((run / "status.json").read_text())
    status["selected_sha256"] = "old-unused-checkpoint-hash"
    write_json(run / "status.json", status)
    data = json.loads((run / "data.json").read_text())
    data["table"] = legacy
    write_json(run / "data.json", data)
    before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    first = evaluation.Evaluation(run, cache)
    a, meta = first.generate("cfg", None, 4)
    second = evaluation.Evaluation(run, cache)
    b, repeated = second.generate("cfg", None, 4)
    assert first.output != second.output
    assert meta["generation_seed"] == repeated["generation_seed"]
    assert a.equals(b)
    assert not {"implementation", "checkpoint"} & meta["request"].keys()
    assert "selected_sha256" not in meta["training_runs"][0]
    assert before == {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        evaluation.Evaluation(run, cache, output_dir=first.output)
    # A valid-looking but changed partition is still rejected by its file checksum.
    changed = copy.deepcopy(checkpoint["splits"])
    changed["train"][0], changed["test"][0] = changed["test"][0], changed["train"][0]
    np.savez_compressed(run / "split.npz", **changed)
    with pytest.raises(ValueError, match="split checksum"):
        evaluation.Evaluation(run, cache)


def test_table_cache_requires_data_checksums(tmp_path):
    import json

    from diffltm.data.core import TABLE_VERSION, load_table

    root = tmp_path / "tables" / TABLE_VERSION / "acslabour"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"source": {"downloader": "folktables==0.0.12"}})
    )
    with pytest.raises(ValueError, match="file checksums"):
        load_table("acslabour", tmp_path)
