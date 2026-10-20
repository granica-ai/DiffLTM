"""Shared checkpoint restoration, generation requests and long-form scoring."""

import json
from pathlib import Path

import pandas as pd

from .artifacts import (
    atomic_write,
    completed_training,
    evaluation_output,
    file_identity,
    identity,
    json_safe,
    request_seed,
    write_json,
)
from .data.core import load_run_splits, load_table, split_identity, verify_dataset
from .data.preprocessing import imputation_identity, imputed_frame
from .generation.conditioning import Condition
from .generation.rejection import REJECTION_MAX_CANDIDATES, RejectionPool
from .generation.sampling import RECOVERY_POLICY, euler
from .metrics import FIDELITY_KEYS, fidelity, subset
from .scoring import METRIC_SETTINGS, score_rows, unconditional
from .trainer import load_model


class Evaluation:
    def __init__(
        self,
        run_dir,
        cache="data",
        device="cpu",
        guidance=1.0,
        pin=True,
        harpoon_eta=0.2,
        output_dir=None,
    ):
        self.run_dir = Path(run_dir)
        completion = completed_training(self.run_dir)
        self.model, self.checkpoint = load_model(self.run_dir / "selected.pt", device)
        if (
            self.checkpoint["status"] != "complete"
            or self.checkpoint["epoch"] != self.checkpoint["config"]["training"]["epochs"]
            or self.checkpoint["run_id"] != completion["run_id"]
        ):
            raise ValueError("Checkpoint does not certify completed training")
        self.table = load_table(self.checkpoint["table_manifest"]["name"], cache)
        verify_dataset(self.checkpoint["table_manifest"], self.table.manifest, self.table.frame)
        splits = load_run_splits(self.run_dir, self.table, self.checkpoint["splits"])
        self.split_digest = split_identity(splits)
        self.frames = {
            k: imputed_frame(
                self.table.frame.iloc[indices], self.table.schema, self.model.preprocessor.fill
            )
            for k, indices in splits.items()
        }
        self.seed = self.checkpoint["config"]["seed"]
        self.guidance, self.pin = guidance, pin
        self.harpoon_eta = harpoon_eta
        self.output = evaluation_output(self.run_dir / "evaluations", output_dir)
        self.sample_root = self.output / "samples"
        self._rejection_pool = None
        self.identity = dict(
            training_runs=[completion],
            model=self.model.config["representation"] + "_" + self.model.config["objective"],
            metric_settings=dict(METRIC_SETTINGS),
            dataset=self.table.manifest["name"],
            dataset_identity=self.table.manifest["identity"],
            dataset_version=self.table.manifest["version"],
            split_id=self.checkpoint["split_id"],
            run_id=self.checkpoint["run_id"],
            training_seed=self.seed,
            split_seed=self.seed,
            sample_layout="evaluation",
            table_manifest=self.table.manifest,
            split_digest=self.split_digest,
            selected_epoch=self.checkpoint["selected_epoch"],
        )

    def generate(self, method, condition, n):
        if condition is not None:
            condition.validate(self.table.schema)
            for col, value in condition.equality.items():
                if col in self.table.schema.categorical:
                    j = self.table.schema.categorical.index(col)
                    if value not in self.model.preprocessor.maps[j]:
                        raise ValueError(f"Unseen categorical condition {col}={value!r}")
        request = dict(
            method=method,
            predicate=condition.dumps() if condition else None,
            n=n,
            steps=100,
            batch_size=4096,
            guidance=self.guidance if method == "cfg" else None,
            pin=self.pin if method in {"cfg", "smc"} else False,
            recovery_policy=dict(RECOVERY_POLICY) if method != "smc" else None,
        )
        if method == "rejection":
            request["candidate_budget"] = REJECTION_MAX_CANDIDATES
        if method == "harpoon":
            from .generation.harpoon import HARPOON_RECOVERY_POLICY, guidance_policy

            request.update(
                harpoon_eta=self.harpoon_eta,
                guidance_policy=guidance_policy(self.model),
                recovery_policy=dict(HARPOON_RECOVERY_POLICY),
            )
        # Recovery limits govern stopping, not the ordinary sampling stream.
        # Deliberately share random numbers across DiffLTM models for the same
        # training seed and request; compare model differences within each seed.
        seed_request = {k: v for k, v in request.items() if k != "recovery_policy"}
        seed = request_seed(self.seed, seed_request, "generation")
        key = identity(request)
        root = self.sample_root / key
        path = root / "samples.parquet"
        meta = root / "manifest.json"
        write_json(
            meta, dict(status="pending", request=request, generation_seed=seed, **self.identity)
        )
        try:
            if method == "unconditional":
                if condition is not None:
                    raise ValueError("Unconditional generation cannot take a predicate")
                frame, diagnostics = euler(self.model, n, seed)
            elif method == "cfg":
                frame, diagnostics = euler(self.model, n, seed, condition, self.guidance, self.pin)
            elif method == "smc":
                from .generation.smc import sample

                frame, diagnostics = sample(self.model, n, seed, condition, self.pin)
            elif method == "harpoon":
                from .generation.harpoon import sample

                frame, diagnostics = sample(self.model, n, seed, condition, self.harpoon_eta)
            elif method == "rejection":
                if condition is None:
                    raise ValueError("Rejection requires a predicate")
                if self._rejection_pool is None:
                    # The same paired-randomness policy applies to candidate pools.
                    self._rejection_pool = RejectionPool(
                        lambda size, chunk_seed: euler(self.model, size, chunk_seed),
                        request_seed(self.seed, "unconditional", "rejection-pool"),
                        directory=self.output,
                    )
                frame, diagnostics = self._rejection_pool.get(condition, n)
            else:
                raise ValueError(f"Unknown method {method}")
            atomic_write(path, lambda p: frame.to_parquet(p, index=False))
            manifest = dict(
                status="complete",
                request=request,
                generation_seed=seed,
                **self.identity,
                diagnostics=diagnostics,
                sha256=file_identity(path),
                sample_key=key,
            )
            write_json(meta, manifest)
            return frame, manifest
        except BaseException as exc:
            write_json(
                meta,
                dict(
                    status="failed",
                    request=request,
                    generation_seed=seed,
                    error=repr(exc),
                    diagnostics=dict(recovery=getattr(exc, "diagnostics", None)),
                    **self.identity,
                ),
            )
            raise

    def score_rows(self, scores, **cell):
        return score_rows(scores, self.identity, **cell)

    def unconditional(self):
        train, test = self.frames["train"], self.frames["test"]
        samples, manifest = self.generate("unconditional", None, len(train))
        rows, extra = unconditional(
            train,
            test,
            samples,
            self.table.schema,
            self.identity,
            manifest["generation_seed"],
        )
        return rows, dict(**extra, sample_key=manifest["sample_key"])

    def slices(self, path, method):
        path = Path(path)
        manifest = json.loads(path.with_name("manifest.json").read_text())
        if manifest["status"] != "complete" or manifest["csv_sha256"] != file_identity(path):
            raise ValueError("Slices must be complete and frozen before evaluation")
        verify_dataset(manifest["dataset"], self.table.manifest, self.table.frame)
        frame = pd.read_csv(path)
        # Older runs may still use a per-run Mincer battery.
        if manifest["task"] == "mincer":
            if (
                manifest["split_digest"] != self.split_digest
                if "split_digest" in manifest
                else manifest["split_id"] != self.checkpoint["split_id"]
            ):
                raise ValueError("Mincer slice/checkpoint split mismatch")
            return frame, manifest
        if manifest.get("slice_seed") != 0:
            raise ValueError(
                "Final conditional evaluation requires a shared reference-seed battery"
            )
        audit_path = path.with_name("run_counts.csv")
        if file_identity(audit_path) != manifest["run_counts_sha256"]:
            raise ValueError("Slice membership audit checksum mismatch")
        audit = pd.read_csv(audit_path)
        audit = audit.loc[audit.training_seed.eq(self.seed)]
        fill_id = imputation_identity(self.table.schema, self.model.preprocessor.fill)
        saved_split = manifest.get("runs", {}).get(str(self.seed), {}).get("split_digest")
        split_matches = (
            saved_split == self.split_digest
            if saved_split is not None
            else audit.split_id.eq(self.checkpoint["split_id"]).all()
        )
        if (
            set(audit.slice_id) != set(frame.slice_id)
            or not split_matches
            or not audit.imputation_id.eq(fill_id).all()
            or (audit.test_count < 100).any()
        ):
            raise ValueError("Slice audit does not match the run's split/imputation/counts")
        frame = frame.merge(audit, on="slice_id", validate="one_to_one")
        for row in frame.itertuples():
            condition = Condition.loads(row.predicate)
            if row.slice_id != identity([row.task, condition.dumps()])[:16]:
                raise ValueError("Shared predicate identity mismatch")
            for key, values in self.frames.items():
                if int(condition.mask(values).sum()) != getattr(row, key + "_count"):
                    raise ValueError("Frozen slice run count mismatch")
        return frame, manifest

    def conditional(self, path, method):
        slices, manifest = self.slices(path, method)
        if method == "harpoon" and manifest["task"] not in {"equality", "range", "linear"}:
            raise ValueError("Harpoon fidelity supports equality/range/linear tasks only")
        rows = []
        for row in slices.itertuples():
            condition = Condition.loads(row.predicate)
            train = self.frames["train"].loc[condition.mask(self.frames["train"])]
            test = self.frames["test"].loc[condition.mask(self.frames["test"])]
            if (len(train), len(test)) != (row.train_count, row.test_count):
                raise ValueError("Frozen slice count mismatch")
            seed = request_seed(self.seed, row.slice_id, "scoring")
            if method == "ideal":
                n = min(len(train), len(test), 100000)
                samples = subset(train, n, seed)
                sample_manifest = dict(generation_seed=None, diagnostics={})
            else:
                n = min(len(test) if method == "rejection" else len(train), 100000)
                samples, sample_manifest = self.generate(method, condition, n)
            exhausted = sample_manifest["diagnostics"].get("budget_exhausted", False)
            matched = 0 if exhausted else min(len(samples), len(test), 100000)
            scores = (
                {metric: float("nan") for metric in FIDELITY_KEYS}
                if exhausted
                else fidelity(test, samples, self.table.schema, seed, cap=100000)
            )
            cell_rows = self.score_rows(
                scores,
                task=row.task,
                method=method,
                slice=row.slice_id,
                bucket=row.bucket,
                scoring_seed=seed,
                generation_seed=sample_manifest["generation_seed"],
                requested=n,
                returned=len(samples),
                n_reference=len(test),
                n_scored=matched,
                train_fraction=row.train_fraction,
                reference_train_fraction=row.reference_train_fraction,
                slice_csv_sha256=manifest["csv_sha256"],
                pre_pin_adherence=sample_manifest["diagnostics"].get("pre_pin_adherence"),
                post_pin_adherence=sample_manifest["diagnostics"].get("post_pin_adherence"),
                adherence=sample_manifest["diagnostics"].get("adherence"),
                budget_exhausted=exhausted,
                candidate_budget=sample_manifest["diagnostics"].get("candidate_budget"),
                pool_extent=sample_manifest["diagnostics"].get("pool_extent"),
            )
            if condition.predicate:
                for cell in cell_rows:
                    cell["slice_name"] = condition.predicate.rsplit(":", 2)[1]
            if exhausted:
                for cell in cell_rows:
                    cell["omission_reason"] = (
                        "Rejection candidate budget exhausted before request filled"
                    )
            rows.extend(cell_rows)
        return rows, dict(slice_manifest=manifest)

    def economic(self, family, method, slices_path=None):
        if method == "harpoon":
            raise ValueError("Harpoon is a conditional fidelity baseline, not an economics method")
        from .economics.experiments import eligible_mincer, observed_cell, requests

        slice_frame, slice_manifest = None, None
        if family == "mincer" and method != "unconditional":
            slice_frame, slice_manifest = self.slices(slices_path, method)
        slice_ids = set(slice_frame.slice_id) if slice_frame is not None else set()
        requests_list = requests(self.table, family, self.frames["test"], slice_frame, method)
        cells = []
        for label, condition, n in requests_list:
            if n < 1:
                cells.append(dict(cell=label, requested=n, status="unsupported"))
                continue
            if family == "mincer" and method == "ideal":
                train = self.frames["train"]
                train = eligible_mincer(train.loc[condition.mask(train)])
                manifest = dict(
                    sample_key=None,
                    generation_seed=request_seed(self.seed, label, "ideal-economic"),
                )
                returned = min(n, len(train))
            else:
                generation_condition = (
                    None if family == "mincer" and method == "unconditional" else condition
                )
                samples, manifest = self.generate(method, generation_condition, n)
                returned = len(samples)
            real = self.frames["test"].loc[condition.mask(self.frames["test"])]
            if family == "mincer":
                real = eligible_mincer(real)
            observed = (
                observed_cell(self.table, condition) if family in {"cps", "reparam"} else real
            )
            source = "reference.parquet" if family in {"cps", "reparam"} else "model_test"
            cells.append(
                dict(
                    cell=label,
                    condition=condition.dumps(),
                    requested=n,
                    returned=returned,
                    test_count=len(real),
                    observed_reference_count=len(observed),
                    observed_reference_source=source,
                    observed_reference_identity=(
                        self.table.manifest["files"][source]
                        if source == "reference.parquet"
                        else self.checkpoint["split_id"]
                    ),
                    n_scored=None,
                    oracle_kind="analytic" if family == "reparam" else None,
                    sample_key=manifest["sample_key"],
                    status="complete",
                    generation_seed=manifest["generation_seed"],
                    scoring_seed=request_seed(self.seed, label, "economic-scoring"),
                    **self.identity,
                    method=method,
                    family=family,
                )
            )
            if family == "mincer":
                cells[-1]["cell_type"] = "rarity" if label in slice_ids else "diagnostic"
                if label in slice_ids:
                    cells[-1]["slice_csv_sha256"] = slice_manifest["csv_sha256"]
        return cells


def evaluate(cfg):
    session = Evaluation(
        cfg["run_dir"],
        cfg["cache"],
        cfg["device"],
        cfg["guidance"],
        cfg["pin"],
        cfg.get("harpoon_eta", 0.2),
        cfg.get("output_dir"),
    )
    output = session.output
    write_json(output / "manifest.json", dict(status="pending", config=cfg, **session.identity))
    try:
        if cfg["task"] == "unconditional":
            rows, extra = session.unconditional()
        elif cfg["task"] == "conditional":
            rows, extra = session.conditional(cfg["slices"], cfg["method"])
        elif cfg["task"] == "economics":
            rows = session.economic(cfg["family"], cfg["method"], cfg["slices"])
            extra = dict(cells=rows)
        elif cfg["task"] == "sample":
            condition = Condition(**cfg["condition"]) if cfg["condition"] else None
            _, extra = session.generate(cfg["method"], condition, cfg["n"])
            rows = []
        else:
            raise ValueError(cfg["task"])
        if cfg["task"] == "conditional" and extra["slice_manifest"]["task"] == "arbit_num":
            extra["scores"] = rows
        atomic_write(output / "scores.csv", lambda p: pd.DataFrame(rows).to_csv(p, index=False))
        write_json(
            output / "manifest.json",
            json_safe(
                {
                    **session.identity,
                    **extra,
                    "status": "complete",
                    "config": cfg,
                    "files": {"scores.csv": file_identity(output / "scores.csv")},
                }
            ),
        )
    except BaseException as exc:
        write_json(
            output / "manifest.json",
            dict(status="failed", config=cfg, error=repr(exc), **session.identity),
        )
        raise
    finally:
        if session._rejection_pool is not None:
            session._rejection_pool.close()
    print(output)
    return output
