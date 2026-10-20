"""Postprocess saved economic requests without a model loader or generator."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from diffltm.artifacts import (
    atomic_write,
    file_identity,
    json_safe,
    verify_result_training,
    write_json,
)
from diffltm.data.core import load_run_splits, load_table, split_identity
from diffltm.data.cps import CPI_U_RS
from diffltm.economics.experiments import (
    CPS_SUBGROUPS,
    MINCER_DIAGNOSTICS,
    eligible_mincer,
    empirical_cdfs,
    matched_mincer,
    observed_cell,
    profiles,
    synthetic_summary,
)
from diffltm.generation.conditioning import Condition
from diffltm.metrics import subset
from diffltm.plotting import format_cps_comparisons


def validate_reparam_cell(cell, sample, manifest, frame, table):
    from diffltm.data.synthetic import DEFAULT_CONFIG as dgp
    from diffltm.economics.synthetic_metrics import validate_generated

    request = sample["request"]
    if not any(
        Path(record["run_dir"]).resolve() == Path(manifest["config"]["run_dir"]).resolve()
        and record["run_id"] == manifest["run_id"]
        for record in manifest["training_runs"]
    ):
        raise ValueError("Reparameterization checkpoint lacks matching completion evidence")
    for key in (
        "dataset",
        "dataset_identity",
        "run_id",
        "split_id",
        "training_seed",
        "model",
    ):
        if cell[key] != manifest[key] or sample[key] != manifest[key]:
            raise ValueError(f"Reparameterization cell identity mismatch: {key}")
    period = int(cell["cell"])
    expected = Condition(equality={"S": str(dgp.treated_unit), "T": str(period)})
    if (
        period not in range(len(dgp.mu_t))
        or Condition.loads(cell["condition"]) != expected
        or request["predicate"] != cell["condition"]
        or request["method"] != cell["method"]
        or cell["method"] != manifest["config"]["method"]
        or cell["sample_key"] != sample["sample_key"]
        or cell["generation_seed"] != sample["generation_seed"]
    ):
        raise ValueError("Reparameterization cell does not match its saved generation request")
    if not (cell["requested"] == cell["returned"] == request["n"] == 100000):
        raise ValueError("Reparameterization cells require requested = returned = 100000")
    if not set(table.schema.columns) <= set(frame.columns):
        raise ValueError("Missing generated reparameterization columns")
    validate_generated(frame, period, dgp, 100000)


def reparam_control(table):
    from diffltm.data.synthetic import DEFAULT_CONFIG as dgp
    from diffltm.economics.synthetic_metrics import fit_sc, q50

    target = table.schema.target
    path, weights = fit_sc(table.frame, value_col=target)
    pre = [t for t in range(len(dgp.mu_t)) if t not in dgp.post_periods]
    medians = table.frame.groupby(["T", "S"])[target].agg(q50).unstack()
    treated = medians.loc[list(map(str, pre)), str(dgp.treated_unit)].to_numpy()
    donors = list(weights)
    donor_medians = medians.loc[list(map(str, pre)), donors].to_numpy()
    lower, upper = donor_medians.min(axis=1), donor_medians.max(axis=1)
    return dict(
        target=target,
        units=target,
        M=table.manifest["M"],
        dataset=table.manifest["name"],
        dataset_identity=table.manifest["identity"],
        source=table.manifest["source"],
        retained_panel_sha256=table.manifest["files"]["table.parquet"],
        dgp=table.manifest["dgp"],
        treated_unit=dgp.treated_unit,
        donor_ids=donors,
        weights=weights,
        path=path.tolist(),
        pre_periods=pre,
        treated_pre_medians=treated.tolist(),
        pre_fit_rmse=float(np.sqrt(np.mean((path[pre] - treated) ** 2))),
        donor_pre_min=lower.tolist(),
        donor_pre_max=upper.tolist(),
        treated_pre_outside_donor_range=((treated < lower) | (treated > upper)).tolist(),
    )


def analyze(family, paths, output, cache="data"):
    contexts = []
    training_runs = {}
    for root in map(Path, paths):
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest["status"] != "complete"
            or manifest["config"]["task"] != "economics"
            or manifest["config"]["family"] != family
        ):
            raise ValueError("Expected complete matching economics request manifest")
        verify_result_training(manifest)
        for record in manifest["training_runs"]:
            training_runs[record["run_dir"]] = record
        table = load_table(manifest["dataset"], cache)
        run = Path(manifest["config"]["run_dir"])
        splits = load_run_splits(run, table)
        test = table.frame.iloc[splits["test"]]
        train = table.frame.iloc[splits["train"]] if family == "mincer" else None
        for cell in manifest["cells"]:
            if cell["status"] != "complete":
                raise ValueError(f"Unsupported/incomplete economic generation cell: {cell}")
            if family == "mincer" and cell["method"] == "ideal":
                condition = Condition.loads(cell["condition"])
                frame = eligible_mincer(train.loc[condition.mask(train)])
                frame = frame.sample(
                    n=cell["returned"], replace=False, random_state=cell["generation_seed"]
                )
            else:
                sample_root = (
                    (root if manifest.get("sample_layout") == "evaluation" else run)
                    / "samples"
                    / cell["sample_key"]
                )
                sample_meta = json.loads((sample_root / "manifest.json").read_text())
                if (
                    sample_meta["status"] != "complete"
                    or file_identity(sample_root / "samples.parquet") != sample_meta["sha256"]
                ):
                    raise ValueError("Incomplete/corrupt samples")
                frame = pd.read_parquet(sample_root / "samples.parquet")
            if family == "reparam":
                validate_reparam_cell(cell, sample_meta, manifest, frame, table)
            cell = dict(
                cell,
                dataset_identity=table.manifest["identity"],
                dataset_version="v1",
                split_id=split_identity(splits),
            )
            contexts.append((cell, frame, table, test, manifest))
    if not contexts:
        raise ValueError("No completed economic cells")
    keys = [
        (c["dataset_identity"], c["training_seed"], c["cell"], c["method"]) for c, *_ in contexts
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate run/cell/method result")
    rows = []
    fits = []
    cdfs = []
    thresholds = []
    if family == "mincer":
        groups = {}
        for item in contexts:
            c = item[0]
            groups.setdefault(
                (c["dataset_identity"], c["split_id"], c["training_seed"], c["cell"]), []
            ).append(item)
        for _, items in groups.items():
            first, _, _, test, manifest = items[0]
            condition = Condition.loads(first["condition"])
            if any(c["condition"] != first["condition"] for c, *_ in items):
                raise ValueError("Mincer methods must use the same predicate")
            real = test.loc[condition.mask(test)]
            generated = {c["method"]: frame for c, frame, *_ in items}
            result, fit = matched_mincer(real, generated, first["scoring_seed"])
            cell_type = first.get(
                "cell_type", "diagnostic" if first["cell"] in MINCER_DIAGNOSTICS else "rarity"
            )
            if cell_type == "rarity":
                slices_path = Path(manifest["config"]["slices"])
                slices_identity = file_identity(slices_path)
                slice_meta = json.loads(slices_path.with_name("manifest.json").read_text())
                if slice_meta["csv_sha256"] != slices_identity or any(
                    c.get("slice_csv_sha256", slices_identity) != slices_identity for c, *_ in items
                ):
                    raise ValueError("Mincer slice checksum mismatch")
                rarity = pd.read_csv(slices_path).set_index("slice_id").loc[first["cell"]]
                if Condition.loads(rarity["predicate"]).dumps() != condition.dumps():
                    raise ValueError("Mincer slice predicate mismatch")
                bucket = rarity["bucket"]
                fraction = rarity.get("reference_train_fraction", rarity.get("train_fraction"))
            else:
                bucket = fraction = slices_identity = None
            fits.append(
                dict(cell=first["cell"], cell_type=cell_type, seed=first["training_seed"], **fit)
            )
            if not result:
                result = [
                    dict(
                        method=method,
                        coefficient_error=float("nan"),
                        fitted=fit["fitted"],
                        status="undefined",
                        omission_reason=fit.get("reason", "Unsupported Mincer fit"),
                    )
                    for method in generated
                ]
            cells = {c["method"]: c for c, *_ in items}
            for r in result:
                cell = cells[r["method"]]
                rows.append(
                    dict(
                        dataset=first["dataset"],
                        dataset_identity=first["dataset_identity"],
                        seed=first["training_seed"],
                        cell=first["cell"],
                        cell_type=cell_type,
                        bucket=bucket,
                        reference_train_fraction=fraction,
                        slice_csv_sha256=slices_identity,
                        requested=cell["requested"],
                        returned=cell["returned"],
                        test_count=cell["test_count"],
                        real_eligible=fit["real_eligible"],
                        real_capped=fit["real_capped"],
                        generated_eligible=fit["generated_eligible"][r["method"]],
                        n_scored=fit["fitted"],
                        units="coefficient_squared_error",
                        **r,
                    )
                )
    else:
        control_cache = {}
        for cell, frame, table, test, _ in contexts:
            base = dict(
                dataset=cell["dataset"],
                dataset_identity=cell["dataset_identity"],
                seed=cell["training_seed"],
                method=cell["method"],
                cell=cell["cell"],
                requested=cell["requested"],
                count=len(frame),
                test_count=cell["test_count"],
                observed_reference_count=cell["observed_reference_count"],
                observed_reference_source=cell["observed_reference_source"],
                observed_reference_identity=cell["observed_reference_identity"],
                n_scored=len(frame),
            )
            condition = Condition.loads(cell["condition"])
            if family == "profiles":
                for method, f in [
                    (cell["method"], frame),
                    ("real", test.loc[condition.mask(test)]),
                ]:
                    for r in profiles(f):
                        rows.append(dict(base, **r, method=method))
            elif family == "reparam":
                period = int(cell["cell"])
                target = table.schema.target
                if cell["dataset"] not in control_cache:
                    control_cache[cell["dataset"]] = reparam_control(table)
                control = control_cache[cell["dataset"]]
                result = synthetic_summary(frame, period, table)
                rows.append(
                    dict(
                        base,
                        **result,
                        sc=control["path"][period],
                        units=target,
                        M=table.manifest["M"],
                        oracle_kind="analytic",
                    )
                )
            elif family == "cps":
                from diffltm.economics.cps_control import fit_sc, nominal

                state = table.manifest["held_state"]
                year = int(condition.equality["YEAR"])
                reference = observed_cell(table, condition)
                is_path = len(condition.equality) == 2
                if is_path:
                    if state not in control_cache:
                        control_cache[state] = fit_sc(table.reference, state, 0.1)
                    control = control_cache[state]
                    value = float(frame.LOG_REAL_HOURWAGE.quantile(0.1))
                    rows.append(
                        dict(
                            base,
                            year=year,
                            quantile=0.1,
                            log_value=value,
                            value=nominal(value, year),
                            real=control["y_treated_nominal"][str(year)],
                            sc=control["y_synthetic_nominal"][str(year)],
                            units="nominal_hourly_dollars",
                        )
                    )
                    continue
                label = next(
                    label
                    for label, (col, code) in CPS_SUBGROUPS.items()
                    if condition.equality.get(col) == code
                )
                real_wage = np.exp(reference.LOG_REAL_HOURWAGE) * CPI_U_RS[year] / 376.5
                syn_wage = np.exp(frame.LOG_REAL_HOURWAGE) * CPI_U_RS[year] / 376.5
                curves = empirical_cdfs(real_wage, syn_wage)
                n = min(len(real_wage), len(syn_wage), 100000)
                real_values = subset(real_wage, n, cell["scoring_seed"])
                syn_values = subset(syn_wage, n, cell["scoring_seed"])
                for k, v in dict(
                    dataset=cell["dataset"],
                    dataset_identity=cell["dataset_identity"],
                    seed=cell["training_seed"],
                    year=year,
                    subgroup=label,
                    generator=cell["method"],
                    units="nominal_hourly_dollars",
                    test_count=cell["test_count"],
                    observed_reference_count=len(reference),
                    observed_reference_source=cell["observed_reference_source"],
                    observed_reference_identity=cell["observed_reference_identity"],
                    generated_count=len(frame),
                    scoring_seed=cell["scoring_seed"],
                    n_scored=n,
                ).items():
                    curves[k] = v
                cdfs.append(curves)
                thresholds.append(
                    dict(
                        base,
                        year=year,
                        subgroup=label,
                        generated_count=len(frame),
                        scoring_seed=cell["scoring_seed"],
                        n_scored=n,
                        ks=float(ks_2samp(real_values, syn_values).statistic) if n else np.nan,
                        real_sub11=float((real_values < 11).mean()),
                        generated_sub11=float((syn_values < 11).mean()),
                    )
                )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    # A real profile is shared by methods; require exact equality before collapsing it.
    if family == "profiles" and not summary.empty:
        summary = summary.drop_duplicates()
    atomic_write(output / "summary.csv", lambda p: summary.to_csv(p, index=False))
    if family == "reparam":
        write_json(output / "synthetic_controls.json", control_cache)
    if cdfs:
        atomic_write(
            output / "cdfs.csv", lambda p: pd.concat(cdfs, ignore_index=True).to_csv(p, index=False)
        )
    if thresholds:
        per_year = pd.DataFrame(thresholds)
        atomic_write(output / "sub11.csv", lambda p: per_year.to_csv(p, index=False))
        comparisons, comparison_summary = cps_comparisons(per_year)
        atomic_write(output / "comparisons.csv", lambda p: comparisons.to_csv(p, index=False))
        atomic_write(
            output / "comparison_summary.csv", lambda p: comparison_summary.to_csv(p, index=False)
        )
    write_json(
        output / "manifest.json",
        json_safe(
            dict(
                status="complete",
                training_runs=list(training_runs.values()),
                family=family,
                inputs=[
                    dict(path=str(p), sha256=file_identity(Path(p) / "manifest.json"))
                    for p in paths
                ],
                fits=fits,
                datasets=sorted({c["dataset"] for c, *_ in contexts}),
                summary_sha256=file_identity(output / "summary.csv"),
                files={
                    name: file_identity(output / name)
                    for name in [
                        "summary.csv",
                        "cdfs.csv",
                        "sub11.csv",
                        "comparisons.csv",
                        "comparison_summary.csv",
                        "synthetic_controls.json",
                    ]
                    if (output / name).exists()
                },
            )
        ),
    )
    return summary


def cps_comparisons(per_year):
    keys = ["dataset", "dataset_identity", "method", "seed", "subgroup"]
    if per_year.duplicated(keys + ["year"]).any():
        raise ValueError("Duplicate CPS subgroup/year")
    if not per_year.groupby(keys).year.apply(lambda y: set(y) == {2016, 2019}).all():
        raise ValueError("CPS subgroup comparisons require both 2016 and 2019")
    columns = [
        "ks",
        "real_sub11",
        "generated_sub11",
        "n_scored",
        "test_count",
        "observed_reference_count",
        "generated_count",
    ]
    wide = per_year.pivot(index=keys, columns="year", values=columns)
    wide.columns = [f"{metric}_{year}" for metric, year in wide.columns]
    wide = wide.reset_index()
    wide["delta_ks"] = wide.ks_2019 - wide.ks_2016
    wide["delta_sub11"] = wide.generated_sub11_2019 - wide.real_sub11_2019
    return wide, format_cps_comparisons(wide)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=["cps", "reparam", "mincer", "profiles"])
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="data")
    args = parser.parse_args()
    analyze(args.family, args.results, args.output, args.cache)


if __name__ == "__main__":
    main()
