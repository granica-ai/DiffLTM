"""Saved-cell contracts, auditable SC fits and the saved-only Figure 4 route."""

import copy
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from diffltm.artifacts import (
    completed_training,
    file_identity,
    identity,
    write_json,
)
from diffltm.data.core import load_table, split_table
from diffltm.economics.synthetic_metrics import q50
from diffltm.evaluation import Evaluation
from diffltm.generation.conditioning import (
    Condition,
)
from diffltm.plotting import plot_reparameterization
from scripts import analyze_econ, plot_results


@pytest.fixture
def saved_cell(tmp_path):
    table = load_table("sc_reparam_quadratic", tmp_path / "data")
    run = tmp_path / "run"
    run.mkdir()
    (run / "selected.pt").write_bytes(b"fixture native checkpoint identity")
    write_json(run / "config.json", dict(training=dict(epochs=1)))
    write_json(
        run / "status.json",
        dict(
            status="complete",
            epoch=1,
            run_id="fixture",
            selected_sha256=file_identity(run / "selected.pt"),
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
        dict(
            table=table.manifest,
            split_id="fixture-split",
            splits_sha256=file_identity(run / "split.npz"),
        ),
    )
    common = dict(
        dataset=table.manifest["name"],
        dataset_identity=table.manifest["identity"],
        checkpoint=file_identity(run / "selected.pt"),
        run_id="fixture",
        split_id="fixture-split",
        training_seed=500,
        model="learnt_logits_ce",
    )
    condition = Condition(equality={"S": "6", "T": "8"}).dumps()
    request = dict(checkpoint=common["checkpoint"], method="cfg", predicate=condition, n=100000)
    key = identity(request)
    frame = pd.DataFrame(
        {
            "U": np.zeros(100000),
            "QUAD_Y": np.full(100000, table.manifest["M"] ** 2),
            "S": "6",
            "T": "8",
        }
    )
    sample = dict(common, status="complete", request=request, sample_key=key, generation_seed=73)
    cell = dict(
        common,
        cell="8",
        method="cfg",
        family="reparam",
        status="complete",
        condition=condition,
        requested=100000,
        returned=100000,
        sample_key=key,
        generation_seed=73,
        test_count=0,
        observed_reference_count=1200,
        observed_reference_source="reference.parquet",
        observed_reference_identity=table.manifest["files"]["reference.parquet"],
    )
    manifest = dict(
        common,
        status="complete",
        config=dict(task="economics", family="reparam", method="cfg", run_dir=str(run)),
        cells=[cell],
        training_runs=[completed_training(run)],
    )
    return run, table, frame, sample, cell, manifest


def save_cell(fixture):
    run, _, frame, sample, _, manifest = fixture
    root = run / "samples" / sample["sample_key"]
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / "samples.parquet", index=False)
    write_json(root / "manifest.json", dict(sample, sha256=file_identity(root / "samples.parquet")))
    evaluation = run / "evaluations/reparam"
    write_json(evaluation / "manifest.json", manifest)
    return evaluation


@pytest.mark.parametrize(
    "field", ["run_id", "dataset_identity", "split_id", "training_seed", "model"]
)
def test_reparam_cell_identity_binding(saved_cell, field):
    _, table, frame, sample, cell, manifest = saved_cell
    cell[field] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        analyze_econ.validate_reparam_cell(cell, sample, manifest, frame, table)


@pytest.mark.parametrize(
    "change",
    [
        "method",
        "predicate",
        "sample_key",
        "generation_seed",
        "requested",
        "returned",
        "request_n",
        "completion",
    ],
)
def test_reparam_saved_request_binding(saved_cell, change):
    _, table, frame, sample, cell, manifest = saved_cell
    if change == "request_checkpoint":
        sample["request"]["checkpoint"] = "wrong"
    elif change == "method":
        sample["request"]["method"] = "smc"
    elif change == "predicate":
        sample["request"]["predicate"] = Condition(equality={"S": "6", "T": "9"}).dumps()
    elif change == "sample_key":
        sample["sample_key"] = "wrong"
    elif change == "generation_seed":
        cell["generation_seed"] += 1
    elif change in ("requested", "returned"):
        cell[change] = 99999
    elif change == "request_n":
        sample["request"]["n"] = 99999
    else:
        manifest["training_runs"][0]["run_id"] = "wrong"
    with pytest.raises(ValueError):
        analyze_econ.validate_reparam_cell(cell, sample, manifest, frame, table)


@pytest.mark.parametrize("change", ["short", "state", "period", "nan", "inf", "missing"])
def test_reparam_rejects_malformed_rows(saved_cell, change):
    _, table, frame, sample, cell, manifest = saved_cell
    if change == "short":
        frame = frame.iloc[:-1]
    elif change == "state":
        frame.loc[0, "S"] = "6.5"
    elif change == "period":
        frame.loc[0, "T"] = "9"
    elif change == "nan":
        frame.loc[0, "QUAD_Y"] = float("nan")
    elif change == "inf":
        frame.loc[0, "U"] = float("inf")
    else:
        frame = frame.drop(columns="U")
    with pytest.raises(ValueError):
        analyze_econ.validate_reparam_cell(cell, sample, manifest, frame, table)


def test_postprocess_single_supplied_cell_retains_off_support_and_saves_sc(
    saved_cell, tmp_path, monkeypatch
):
    _, table, frame, sample, cell, manifest = saved_cell
    # A second supplied period reuses the same target fit; no full seed/grid is required.
    evaluation = save_cell(saved_cell)
    other = copy.deepcopy(cell)
    other.update(cell="9", condition=Condition(equality={"S": "6", "T": "9"}).dumps())
    other_sample = copy.deepcopy(sample)
    other_sample["request"]["predicate"] = other["condition"]
    other["sample_key"] = other_sample["sample_key"] = identity(other_sample["request"])
    manifest["cells"].append(other)
    save_cell((saved_cell[0], table, frame.assign(T="9"), other_sample, other, manifest))
    monkeypatch.setattr(analyze_econ, "load_table", lambda *a: table)
    original = analyze_econ.reparam_control
    calls = []

    def control(t):
        calls.append(t.manifest["identity"])
        return original(t)

    monkeypatch.setattr(analyze_econ, "reparam_control", control)
    out = tmp_path / "summary"
    result = analyze_econ.analyze("reparam", [evaluation], out)
    assert calls == [table.manifest["identity"]]
    assert len(result) == 2 and set(result.seed) == {500}
    assert result["median"].eq(table.manifest["M"] ** 2).all()
    assert result.above_global_support_fraction.eq(1).all()
    assert result.outside_cell_support_fraction.eq(1).all()
    assert result.n_scored.eq(100000).all() and result.test_count.eq(0).all()
    controls = json.loads((out / "synthetic_controls.json").read_text())
    assert set(controls) == {table.manifest["name"]}
    fit = controls[table.manifest["name"]]
    assert set(fit["donor_ids"]) == set(fit["weights"]) and "6" not in fit["donor_ids"]
    assert sum(fit["weights"].values()) == pytest.approx(1)
    assert min(fit["weights"].values()) >= -1e-7
    assert fit["M"] == table.manifest["M"]
    assert fit["retained_panel_sha256"] == table.manifest["files"]["table.parquet"]
    assert sum(fit["treated_pre_outside_donor_range"]) == 7
    treated = table.frame.loc[table.frame.S.eq("6")].groupby("T").QUAD_Y.agg(q50)
    expected_rmse = np.sqrt(
        np.mean([(fit["path"][t] - treated[str(t)]) ** 2 for t in fit["pre_periods"]])
    )
    assert fit["pre_fit_rmse"] == pytest.approx(expected_rmse)
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["files"]["synthetic_controls.json"] == file_identity(
        out / "synthetic_controls.json"
    )


def figure_frame():
    return pd.DataFrame(
        [
            dict(
                dataset="sc_reparam_" + variant,
                dataset_identity=variant + "-content",
                units=units,
                method="cfg",
                seed=seed,
                period=period,
                median=(seed - 500) + 10 + period,
                sc=8 + period,
                oracle=9 + period,
                oracle_kind="analytic",
                M=50.0,
            )
            for variant, units in [("level", "Y"), ("log", "LOG_Y"), ("quadratic", "QUAD_Y")]
            for period in (0, 8, 10)
            for seed in (500, 501, 502)
        ]
    )


def test_combined_figure_uses_saved_native_values_and_fixed_references(tmp_path, monkeypatch):
    frame = figure_frame()
    before = frame.copy(deep=True)
    output = tmp_path / "figure4"
    monkeypatch.setitem(plt.rcParams, "axes.labelsize", 17)
    caller_style = dict(plt.rcParams)
    fig, axes = plot_reparameterization(frame, output=output)
    assert dict(plt.rcParams) == caller_style
    np.testing.assert_allclose(fig.get_size_inches(), (10, 3.2))
    assert len(axes) == 3 and len(fig.legends) == 1
    assert all(len(ax.patches) == 1 for ax in axes)
    assert [ax.get_title() for ax in axes] == ["Y", "log(Y)", "Y*(M-Y)"]
    assert [ax.get_ylabel() for ax in axes] == ["Median (Y)", "Median (logY)", "Median (Z)"]
    legend = fig.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "Oracle median",
        "Synthetic control",
        "DiffLTM-CFG",
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        assert ax.title.get_fontweight() == "bold" and ax.title.get_fontsize() == 10
        assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 12
        assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
        np.testing.assert_array_equal(ax.get_xticks(), range(0, 11, 2))
        assert ax.title.get_window_extent(renderer).y1 < legend.get_window_extent(renderer).y0
        cfg = ax.containers[0]
        assert cfg.get_label() == "DiffLTM-CFG"
        assert cfg.lines[0].get_markeredgewidth() == 0.6
        assert all(cap.get_markeredgewidth() == 0.7 for cap in cfg.lines[1])
        assert all(np.all(bars.get_linewidth() == 0.7) for bars in cfg.lines[2])
    with Image.open(output.with_suffix(".png")) as png:
        np.testing.assert_allclose(png.info["dpi"], (300, 300), atol=0.01)
    values = pd.read_csv(output.with_suffix(".csv"))
    generated = values.loc[values.method.eq("cfg")]
    np.testing.assert_allclose(generated["mean"], 11 + generated.period)
    np.testing.assert_allclose(generated.se, 1 / np.sqrt(3))
    assert generated.n_runs.eq(3).all()
    assert values.loc[~values.method.eq("cfg"), "se"].isna().all()
    assert values.loc[~values.method.eq("cfg"), "n_runs"].isna().all()
    for method, intercept in [("sc", 8), ("oracle", 9)]:
        reference = values.loc[values.method.eq(method)]
        np.testing.assert_allclose(reference["mean"], intercept + reference.period)
    pd.testing.assert_frame_equal(frame, before)
    for suffix in (".pdf", ".png", ".csv", ".json"):
        assert output.with_suffix(suffix).is_file()
    plt.close(fig)
    # Supplied cells/one seed suffice; undefined SE is retained, not fabricated.
    fig, axes = plot_reparameterization(frame.loc[frame.seed.eq(500)], xlim=(2, 8), ylim=(5, 25))
    for ax in axes:
        assert ax.get_xlim() == (2, 8) and ax.get_ylim() == (5, 25)
    plt.close(fig)


@pytest.mark.parametrize("change", ["units", "dataset_identity", "M", "oracle", "duplicate"])
def test_combined_figure_rejects_mixed_identities(change):
    frame = figure_frame()
    if change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame.loc[0, change] = "wrong" if change in ("units", "dataset_identity") else 123.0
    with pytest.raises(ValueError):
        plot_reparameterization(frame)
    plt.close("all")


@pytest.mark.parametrize("figsize", [None, [8, 3]])
def test_figure4_only_inventory_route_does_not_require_other_outputs_or_seed_grid(
    tmp_path, monkeypatch, figsize
):

    source = Path(__file__).resolve().parents[1]
    inventory = pd.read_csv(source / "paper/experiments.csv").fillna("")
    row = inventory.loc[inventory.output.eq("figure4")].iloc[0]
    assert json.loads(row.figsize) == [10, 3.2]
    if figsize is not None:
        inventory.loc[inventory.output.eq("figure4"), "figsize"] = json.dumps(figsize)
    assert row.plotting_function == "plot_reparameterization"
    assert "100000" in row.metric_settings and "MLE" not in row.metric_settings
    frame = figure_frame().loc[lambda f: f.seed.eq(500)]
    paths = []
    for dataset, part in frame.groupby("dataset"):
        root = tmp_path / dataset
        root.mkdir()
        path = root / "summary.csv"
        part.to_csv(path, index=False)
        paths.append(str(path))
        write_json(
            root / "manifest.json",
            dict(status="complete", files={"summary.csv": file_identity(path)}),
        )
    inventory.loc[inventory.output.eq("figure4"), "results"] = ";".join(paths)
    path = tmp_path / "inventory.csv"
    inventory.to_csv(path, index=False)
    monkeypatch.setattr(plot_results, "verify_result_training", lambda m: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_results",
            "--manifest",
            str(path),
            "--output",
            str(tmp_path / "figures"),
            "--only",
            "figure4",
        ],
    )
    closed = []
    close = plt.close

    def record_close(fig):
        closed.append(fig)
        close(fig)

    monkeypatch.setattr(plt, "close", record_close)
    plot_results.main()
    assert len(closed) == 1 and not plt.fignum_exists(closed[0].number)
    np.testing.assert_allclose(closed[0].get_size_inches(), figsize or [10, 3.2])
    assert (tmp_path / "figures/figure4.pdf").exists()
    provenance = json.loads((tmp_path / "figures/figure4.inputs.json").read_text())
    assert len(provenance["inputs"]) == 3 and all(
        "manifest_sha256" in r for r in provenance["inputs"]
    )
    plt.close("all")


def test_economic_heldout_counts_identify_empirical_and_analytic_references(tmp_path, monkeypatch):
    from diffltm.data.core import load_table

    table = load_table("sc_reparam_level", tmp_path)
    ev = object.__new__(Evaluation)
    ev.table = table
    ev.frames = {"test": table.frame.iloc[:100]}
    ev.seed = 500
    ev.identity = dict(dataset="sc_reparam_level")
    ev.checkpoint = {"split_id": "test-split"}
    monkeypatch.setattr(
        ev,
        "generate",
        lambda *args: (pd.DataFrame({"Y": [1, 2]}), dict(sample_key="fixture", generation_seed=1)),
    )
    cells = ev.economic("reparam", "cfg")
    held = next(c for c in cells if c["cell"] == "9")
    assert held["test_count"] == 0
    assert held["observed_reference_count"] == 1200
    assert held["observed_reference_source"] == "reference.parquet"
    assert held["oracle_kind"] == "analytic" and held["n_scored"] is None
    assert held["requested"] == 100000


def test_splits_and_shared_panel(tmp_path, small_table):
    from diffltm.economics.experiments import requests

    first, key = split_table(small_table, 500, tmp_path)
    same, samekey = split_table(small_table, 500, tmp_path)
    other, _ = split_table(small_table, 501, tmp_path)
    assert [len(first[k]) for k in ("train", "validation", "test")] == [84, 12, 24]
    assert key == samekey
    assert all(np.array_equal(first[k], same[k]) for k in first)
    assert not np.array_equal(first["train"], other["train"])
    tables = [load_table("sc_reparam_" + v, tmp_path) for v in ["level", "log", "quadratic"]]
    assert {len(t.frame) for t in tables} == {141600}
    assert len({t.manifest["M"] for t in tables}) == 1
    assert all(tables[0].frame.index.equals(t.frame.index) for t in tables)
    assert tables[0].manifest["M"] == tables[0].frame.Y.max()
    assert np.allclose(
        tables[2].frame.QUAD_Y, tables[0].frame.Y * (tables[0].manifest["M"] - tables[0].frame.Y)
    )
    for t in tables:
        cells = requests(t, "reparam", t.frame)
        assert len(cells) == 11 and all(n == 100000 for _, _, n in cells)
        assert all(c.equality["S"] == "6" for _, c, _ in cells)
        splits, _ = split_table(t, 500, tmp_path)
        assert not ((t.frame.S == "6") & t.frame["T"].isin(["8", "9", "10"])).any()
        if t is tables[0]:
            expected = splits
        else:
            assert all(np.array_equal(expected[k], splits[k]) for k in splits)
    warm = load_table("sc_reparam_level", tmp_path)
    pd.testing.assert_frame_equal(warm.frame, tables[0].frame)


def test_native_log_summary_does_not_invert_or_clip(tmp_path):
    from diffltm.economics.experiments import synthetic_summary

    table = load_table("sc_reparam_log", tmp_path)
    generated = pd.DataFrame({"LOG_Y": [1000.0, 1001.0, 1002.0]})
    result = synthetic_summary(generated, 2, table)
    assert result["median"] == 1001 and generated.LOG_Y.tolist() == [1000.0, 1001.0, 1002.0]


def test_quadratic_invalid_outputs_remain_in_median(tmp_path):
    from diffltm.economics.experiments import synthetic_summary

    table = load_table("sc_reparam_quadratic", tmp_path)
    maximum = table.manifest["M"]
    z = maximum**2 / 4
    generated = pd.DataFrame({"QUAD_Y": [z + 1, z + 2, z + 3]})
    before = generated.copy(deep=True)
    result = synthetic_summary(generated, 0, table)
    assert result["above_global_support_fraction"] == 1 and result["median"] == z + 2
    assert result["outside_cell_support_fraction"] == 1
    pd.testing.assert_frame_equal(before, generated)
