"""Freeze shared predicates and audit membership in each publication split."""

import argparse
import json
import math
import os
import runpy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from diffltm.artifacts import atomic_write, file_identity, identity, request_seed, write_json
from diffltm.data.core import load_table, split_table, verify_dataset
from diffltm.data.preprocessing import fit_numeric_fill, imputation_identity, imputed_frame
from diffltm.generation.conditioning import Condition, standard_to_raw

SLICE_SEED = 0
RUN_SEEDS = (500, 501, 502)
BUCKET_EDGES = (0.03, 0.01, 0.003, 0.001, 0.0)
BUCKET_NAMES = (">3%", "(1%,3%]", "(0.3%,1%]", "(0.1%,0.3%]", "(0,0.1%]")
QUOTA = 10
MAX_DEPTH = 4
RANGE_WIDTH = (0.002, 0.8)
LINEAR_WIDTH = (0.02, 1.0)
MIN_TEST = 100
PROPOSAL_BUDGET = 200000
PILOT_SIZE = 100000
PILOT_SEED_OFFSET = 9000
PILOT_MIN_HITS = 3
PILOT_FACTORS = (0.65, 1.0, 1.5)
LINEAR_GRID = 1e-4
ARBIT_NUM_PATH = Path("slices/acslabour/arbit_num/predicates.py")


class IncompleteBatteryError(RuntimeError):
    pass


def bucket(frequency):
    for i, edge in enumerate(BUCKET_EDGES):
        if frequency > edge:
            return i
    return None


def propose(train, schema, task, rng, stats):
    pool = schema.categorical if task in {"equality", "mincer"} else schema.numerical
    if task == "mincer":
        pool = [c for c in pool if c != "SCHL"]
    if not pool:
        return None
    columns = rng.choice(
        pool, int(rng.integers(1, min(MAX_DEPTH, len(pool)) + 1)), replace=False
    ).tolist()
    anchor = train.iloc[int(rng.integers(len(train)))]
    if task in {"equality", "mincer"}:
        return Condition(equality={c: anchor[c] for c in columns})
    if task == "range":
        ranges = {}
        for c in columns:
            width = np.exp(rng.uniform(*np.log(RANGE_WIDTH)))
            start = rng.uniform(0, 1 - width)
            lo, hi = np.quantile(train[c].to_numpy(float), [start, start + width])
            if c in schema.integer:
                lo, hi = np.floor(lo), np.ceil(hi)
            ranges[c] = (float(lo), float(hi))
        return Condition(ranges=ranges)
    mean, scale, lo, hi = (v[columns].to_numpy(float) for v in stats)
    w = rng.normal(size=len(columns))
    w /= np.linalg.norm(w)
    low, high = (lo - mean) / scale, (hi - mean) / scale
    ends = np.stack([low * w, high * w])
    smin, smax = ends.min(0).sum(), ends.max(0).sum()
    s = float(((anchor[columns].to_numpy(float) - mean) / scale) @ w)
    width = np.exp(rng.uniform(*np.log(LINEAR_WIDTH))) * (smax - smin)
    lower = s - rng.uniform() * width
    upper = lower + width
    a, bounds = standard_to_raw(w, [max(lower, smin), min(upper, smax)], mean, scale)
    # Counts always use the serialized predicate, including rounded coefficients.
    weights = {c: float(f"{v:.10g}") for c, v in zip(columns, a)}
    bounds = (
        float(math.floor(bounds[0] / LINEAR_GRID) * LINEAR_GRID),
        float(math.ceil(bounds[1] / LINEAR_GRID) * LINEAR_GRID),
    )
    return Condition(weights=weights, bounds=bounds)


def arbitrary_conditions():
    digest = file_identity(ARBIT_NUM_PATH)
    functions = runpy.run_path(ARBIT_NUM_PATH)["PREDICATES"]
    return [
        Condition(predicate=f"{ARBIT_NUM_PATH.as_posix()}:{function.__name__}:{digest}")
        for function in functions
    ]


def membership_view(table, seed, cache):
    splits, split_id = split_table(table, seed, cache)
    fill = fit_numeric_fill(table.frame.iloc[splits["train"]], table.schema)
    frames = {k: imputed_frame(table.frame.iloc[v], table.schema, fill) for k, v in splits.items()}
    return frames, dict(
        split_id=split_id,
        split_digest=split_id,
        imputation_id=imputation_identity(table.schema, fill),
        fill=fill.tolist(),
    )


def membership_key(condition, train):
    mask = condition.mask(train)
    if not mask.any() or mask.all():
        return None, mask
    if any(train.loc[mask, c].nunique() < 2 for c in condition.ranges):
        return None, mask
    import hashlib

    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(), mask


def make_battery(table, task, root="slices", cache="data", populations=None):
    if populations is None:
        populations = {
            seed: membership_view(table, seed, cache) for seed in (SLICE_SEED, *RUN_SEEDS)
        }
    frames, reference = populations[SLICE_SEED]
    train = frames["train"]
    pilot = train.sample(
        n=min(PILOT_SIZE, len(train)), replace=False, random_state=SLICE_SEED + PILOT_SEED_OFFSET
    )
    rng = np.random.default_rng(
        request_seed(SLICE_SEED, [table.manifest["name"], task], "slice-proposals")
    )
    num = train[table.schema.numerical]
    stats = (num.mean(), num.std(ddof=0).replace(0, 1), num.min(), num.max())
    fixed = None
    if task == "arbit_num":
        if table.manifest["name"] != "acslabour":
            raise ValueError("Arbitrary numerical predicates are defined only for ACSLabour")
        fixed = arbitrary_conditions()
    filled = [0] * 5
    seen, memberships = set(), set()
    records, audits = [], []
    proposed_depths, accepted_depths = Counter(), Counter()
    proposed = 0
    candidates = fixed or (
        propose(train, table.schema, task, rng, stats) for _ in range(PROPOSAL_BUDGET)
    )
    for proposed, condition in enumerate(candidates, 1):
        if condition is None:
            break
        depth = (
            1
            if fixed is not None
            else len(condition.equality or condition.ranges or condition.weights)
        )
        proposed_depths[depth] += 1
        serialized = condition.dumps()
        if serialized in seen:
            continue
        seen.add(serialized)
        condition = Condition.loads(serialized)
        if fixed is None:
            hits = int(condition.mask(pilot).sum())
            possible = [bucket(min(1.0, factor * hits / len(pilot))) for factor in PILOT_FACTORS]
            if hits < PILOT_MIN_HITS or not any(
                i is not None and filled[i] < QUOTA for i in possible
            ):
                continue
        key, mask = membership_key(condition, train)
        fraction = float(mask.mean())
        b = bucket(fraction)
        if key is None or key in memberships or b is None or (fixed is None and filled[b] >= QUOTA):
            continue
        run_counts = {
            seed: {k: int(condition.mask(f).sum()) for k, f in populations[seed][0].items()}
            for seed in RUN_SEEDS
        }
        if any(counts["test"] < MIN_TEST for counts in run_counts.values()):
            continue
        counts = {k: int(condition.mask(f).sum()) for k, f in frames.items()}
        slice_id = identity([task, serialized])[:16]
        records.append(
            dict(
                slice_id=slice_id,
                task=task,
                predicate=serialized,
                bucket=BUCKET_NAMES[b],
                reference_train_fraction=fraction,
                **{"reference_" + k + "_count": v for k, v in counts.items()},
            )
        )
        for seed, counts in run_counts.items():
            _, run = populations[seed]
            audits.append(
                dict(
                    slice_id=slice_id,
                    training_seed=seed,
                    split_id=run["split_id"],
                    imputation_id=run["imputation_id"],
                    **{k + "_count": v for k, v in counts.items()},
                    train_fraction=counts["train"] / len(populations[seed][0]["train"]),
                )
            )
        memberships.add(key)
        accepted_depths[depth] += 1
        filled[b] += 1
        if fixed is None and all(c == QUOTA for c in filled):
            break
    output = Path(root) / table.manifest["name"] / task
    path = output / "slices.csv"
    columns = [
        "slice_id",
        "task",
        "predicate",
        "bucket",
        "reference_train_fraction",
        "reference_train_count",
        "reference_validation_count",
        "reference_test_count",
    ]
    atomic_write(path, lambda p: pd.DataFrame(records, columns=columns).to_csv(p, index=False))
    audit_columns = [
        "slice_id",
        "training_seed",
        "split_id",
        "imputation_id",
        "train_count",
        "validation_count",
        "test_count",
        "train_fraction",
    ]
    audit_path = output / "run_counts.csv"
    atomic_write(
        audit_path, lambda p: pd.DataFrame(audits, columns=audit_columns).to_csv(p, index=False)
    )
    restored_audit = pd.read_csv(audit_path).set_index(["slice_id", "training_seed"])
    for row in pd.read_csv(path).itertuples():
        condition = Condition.loads(row.predicate)
        for seed, (views, _) in populations.items():
            for k, frame in views.items():
                expected = (
                    getattr(row, "reference_" + k + "_count")
                    if seed == SLICE_SEED
                    else restored_audit.loc[(row.slice_id, seed), k + "_count"]
                )
                if int(condition.mask(frame).sum()) != expected:
                    raise AssertionError("Serialized slice count mismatch")
    complete = len(records) == len(fixed) if fixed is not None else all(c == QUOTA for c in filled)
    write_json(
        output / "manifest.json",
        dict(
            dataset=table.manifest,
            slice_seed=SLICE_SEED,
            reference=reference,
            runs={str(seed): populations[seed][1] for seed in RUN_SEEDS},
            task=task,
            status="complete" if complete else "incomplete",
            proposals=proposed,
            proposed_depths=dict(proposed_depths),
            accepted_depths=dict(accepted_depths),
            unique_proposals=len(seen),
            accepted=sum(filled),
            counts=dict(zip(BUCKET_NAMES, filled)),
            csv_sha256=file_identity(path),
            run_counts_sha256=file_identity(audit_path),
            constants=dict(
                quota=QUOTA,
                proposal_budget=PROPOSAL_BUDGET,
                min_test=MIN_TEST,
                pilot_size=PILOT_SIZE,
                pilot_min_hits=PILOT_MIN_HITS,
                pilot_factors=PILOT_FACTORS,
                max_depth=MAX_DEPTH,
                range_width=RANGE_WIDTH,
                linear_width=LINEAR_WIDTH,
                linear_grid=LINEAR_GRID,
            ),
        ),
    )
    if not complete:
        raise IncompleteBatteryError(f"{task}: {dict(zip(BUCKET_NAMES, filled))}; no rules relaxed")
    return path


def frozen_battery(table, task, root="slices"):
    path = Path(root) / table.manifest["name"] / task / "slices.csv"
    meta = path.with_name("manifest.json")
    if not meta.exists():
        return None
    manifest = json.loads(meta.read_text())
    if manifest["status"] != "complete":
        return None
    fixed = arbitrary_conditions() if task == "arbit_num" else None
    verify_dataset(manifest["dataset"], table.manifest, table.frame)
    if (
        manifest["slice_seed"] != SLICE_SEED
        or set(manifest["runs"]) != set(map(str, RUN_SEEDS))
        or manifest["task"] != task
        or manifest["csv_sha256"] != file_identity(path)
        or manifest["run_counts_sha256"] != file_identity(path.with_name("run_counts.csv"))
    ):
        raise ValueError(f"Frozen battery identity/checksum mismatch: {path}")
    frame = pd.read_csv(path)
    audit = pd.read_csv(path.with_name("run_counts.csv"))
    if fixed is not None and set(frame.predicate) != {condition.dumps() for condition in fixed}:
        raise ValueError(
            f"Frozen battery predicate mismatch: {path}. "
            "Preserve the original predicate definitions or explicitly build a new battery "
            "in a separate output directory."
        )
    expected = len(fixed) if fixed is not None else 5 * QUOTA
    if (
        len(frame) != expected
        or frame.slice_id.duplicated().any()
        or frame.predicate.duplicated().any()
        or (
            fixed is None
            and frame.groupby("bucket").size().to_dict() != dict.fromkeys(BUCKET_NAMES, QUOTA)
        )
        or len(audit) != len(frame) * len(RUN_SEEDS)
        or audit.duplicated(["slice_id", "training_seed"]).any()
        or (audit.test_count < MIN_TEST).any()
    ):
        raise ValueError(f"Frozen battery fails quota/membership contracts: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args()
    cache = os.environ.get("DIFFLTM_DATA", "data")
    table = load_table(args.dataset, cache)
    failures, populations = [], None
    tasks = ["equality", "range", "linear"]
    if table.manifest["name"] == "acslabour":
        tasks.append("arbit_num")
    for task in tasks:
        try:
            path = frozen_battery(table, task)
            if path is None:
                if populations is None:
                    populations = {
                        seed: membership_view(table, seed, cache)
                        for seed in (SLICE_SEED, *RUN_SEEDS)
                    }
                path = make_battery(table, task, populations=populations)
            print(path, flush=True)
        except IncompleteBatteryError as exc:
            failures.append(str(exc))
    if failures:
        raise IncompleteBatteryError("\n".join(failures))


if __name__ == "__main__":
    main()
