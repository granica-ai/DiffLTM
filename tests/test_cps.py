import json

import numpy as np
import pandas as pd
import pytest

from diffltm.artifacts import (
    write_json,
)
from diffltm.data.core import Schema
from diffltm.generation.conditioning import Condition


def test_cps_differences_are_computed_within_seed():
    from scripts import analyze_econ as analyze

    rows = []
    for seed, delta in zip([500, 501, 502], [0.1, 0.2, 0.3]):
        for year in [2016, 2019]:
            rows.append(
                dict(
                    dataset="cps_wage_az",
                    dataset_identity="one",
                    method="cfg",
                    seed=seed,
                    subgroup="healthcare",
                    year=year,
                    ks=0.05 + (delta if year == 2019 else 0),
                    real_sub11=0.15,
                    generated_sub11=0.3,
                    n_scored=10,
                    test_count=0,
                    observed_reference_count=20,
                    generated_count=10,
                )
            )
    wide, summary = analyze.cps_comparisons(pd.DataFrame(rows))
    np.testing.assert_allclose(wide.delta_ks, [0.1, 0.2, 0.3])
    summary = summary.set_index("metric")
    assert summary.loc["delta_ks", "mean"] == pytest.approx(0.2)
    assert summary.loc["delta_ks", "se"] == pytest.approx(0.1 / np.sqrt(3))
    assert summary.loc["delta_sub11", "mean"] == pytest.approx(0.15)


def test_cps_postprocessing_keeps_full_cdfs_and_matched_statistics(tmp_path, monkeypatch):
    from diffltm.artifacts import completed_training, file_identity
    from diffltm.data.core import make_table
    from scripts import analyze_econ as analyze

    raw = pd.DataFrame(
        {
            "LOG_REAL_HOURWAGE": np.log([8.0, 9.0, 12.0, 14.0] * 2),
            "STATE": "AZ",
            "YEAR": [2016] * 4 + [2019] * 4,
            "EDUC": 39,
        }
    )
    schema = Schema(
        ["LOG_REAL_HOURWAGE"], ["STATE", "YEAR", "EDUC"], "LOG_REAL_HOURWAGE", "regression"
    )
    table = make_table(
        raw.iloc[:4], schema, "cps_wage_az", "fixture", reference=raw, held_state="AZ"
    )
    table.frame.to_parquet(tmp_path / "table.parquet")
    table.manifest["files"] = {"table.parquet": file_identity(tmp_path / "table.parquet")}
    monkeypatch.setattr(analyze, "load_table", lambda *args: table)
    roots = []
    seed = 500
    run = tmp_path / str(seed)
    run.mkdir()
    (run / "selected.pt").write_bytes(b"test checkpoint identity")
    write_json(run / "config.json", {"training": {"epochs": 3}})
    write_json(
        run / "status.json",
        dict(
            status="complete",
            epoch=3,
            run_id=str(seed),
        ),
    )
    np.savez(
        run / "split.npz",
        train=np.arange(len(table.frame)),
        validation=np.array([], dtype=int),
        test=np.array([], dtype=int),
    )
    write_json(
        run / "data.json",
        {
            "table": table.manifest,
            "split_id": str(seed),
            "splits_sha256": file_identity(run / "split.npz"),
        },
    )
    cells = []
    for year in [2016, 2019]:
        key = f"hs_{year}"
        sample = run / "samples" / key
        sample.mkdir(parents=True)
        # Deliberately non-adherent education: postprocessing must not filter CFG output.
        frame = pd.DataFrame(
            {"LOG_REAL_HOURWAGE": np.log([7.0, 10.0, 15.0, 16.0, 17.0, 18.0]), "EDUC": "43"}
        )
        frame.to_parquet(sample / "samples.parquet", index=False)
        write_json(
            sample / "manifest.json",
            dict(status="complete", sha256=file_identity(sample / "samples.parquet")),
        )
        cells.append(
            dict(
                dataset="cps_wage_az",
                dataset_identity=table.manifest["identity"],
                training_seed=seed,
                cell=key,
                method="cfg",
                status="complete",
                sample_key=key,
                condition=Condition(
                    equality={"STATE": "AZ", "YEAR": str(year), "EDUC": "39"}
                ).dumps(),
                requested=6,
                test_count=0,
                observed_reference_count=4,
                observed_reference_source="reference.parquet",
                observed_reference_identity="raw",
                scoring_seed=seed,
            )
        )
    root = run / "evaluations/cps"
    write_json(
        root / "manifest.json",
        dict(
            status="complete",
            config=dict(task="economics", family="cps", run_dir=str(run)),
            dataset="cps_wage_az",
            dataset_identity=table.manifest["identity"],
            split_id=str(seed),
            cells=cells,
            training_runs=[completed_training(run)],
        ),
    )
    roots.append(root)
    output = tmp_path / "analysis"
    analyze.analyze("cps", roots, output)
    per_year = pd.read_csv(output / "sub11.csv")
    cdfs = pd.read_csv(output / "cdfs.csv")
    comparisons = pd.read_csv(output / "comparisons.csv")
    assert per_year.n_scored.eq(4).all() and per_year.test_count.eq(0).all()
    assert per_year.observed_reference_count.eq(4).all()
    assert per_year.generated_count.eq(6).all()
    assert len(comparisons) == 1
    for row in per_year.itertuples():
        curves = cdfs.loc[cdfs.seed.eq(row.seed) & cdfs.year.eq(row.year)]
        from scipy.stats import ks_2samp

        from diffltm.data.cps import CPI_U_RS

        assert len(curves) == 10
        assert curves.loc[curves.method.eq("real"), "n"].eq(4).all()
        assert curves.loc[curves.method.eq("generated"), "n"].eq(6).all()
        factor = CPI_U_RS[row.year] / 376.5
        real_values = pd.Series([8.0, 9.0, 12.0, 14.0]) * factor
        syn_values = (pd.Series([7.0, 10.0, 15.0, 16.0, 17.0, 18.0]) * factor).sample(
            n=4, random_state=row.scoring_seed
        )
        assert row.real_sub11 == pytest.approx(real_values.lt(11).mean())
        assert row.generated_sub11 == pytest.approx(syn_values.lt(11).mean())
        assert row.ks == pytest.approx(ks_2samp(real_values, syn_values).statistic)
    verify = json.loads((output / "manifest.json").read_text())
    assert len(verify["training_runs"]) == 1
    assert "comparisons.csv" in verify["files"] and "comparison_summary.csv" in verify["files"]
