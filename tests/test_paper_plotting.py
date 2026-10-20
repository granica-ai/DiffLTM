"""Shared publication styling must not change the saved scientific summaries."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from diffltm.plotting import PALETTE, plot_cdfs, plot_paths, plot_rarity


def assert_paper_layout(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = fig.legends[0].get_window_extent(renderer)
    assert legend.x0 >= 0 and legend.x1 <= fig.bbox.width
    for ax in axes:
        assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
        assert ax.title.get_fontweight() == "bold"
        assert ax.title.get_window_extent(renderer).y1 < legend.y0
        assert any(line.get_visible() for line in ax.get_ygridlines())
        assert not any(line.get_visible() for line in ax.get_xgridlines())


def test_cps_combined_paths_preserve_individual_summaries_and_style(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [
            dict(
                dataset=dataset,
                dataset_identity=dataset,
                units="nominal_hourly_dollars",
                seed=seed,
                method="cfg",
                year=year,
                value=8 + year / 1000 + (seed - 500),
                real=9 + year / 1000,
                sc=8.5 + year / 1000,
            )
            for dataset in ["cps_wage_az", "cps_wage_pa"]
            for seed in [500, 501]
            for year in range(2010, 2020)
        ]
    )
    before = frame.copy(deep=True)
    monkeypatch.setitem(plt.rcParams, "axes.labelsize", 17)
    caller_style = dict(plt.rcParams)
    output = tmp_path / "combined"
    fig, axes = plot_paths(
        frame, dataset="cps_wage_az;cps_wage_pa", units="nominal_hourly_dollars", output=output
    )
    assert dict(plt.rcParams) == caller_style
    assert_paper_layout(fig, axes)
    assert [ax.get_title() for ax in axes] == ["AZ", "PA"]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "Observed",
        "Synthetic control",
        "DiffLTM-CFG",
    ]
    np.testing.assert_allclose(fig.get_size_inches(), [8, 3.2])
    for ax in axes:
        series = {line.get_label(): line for line in ax.lines}
        assert series["Synthetic control"].get_color() == PALETTE["sc"]
        assert series["Synthetic control"].get_linestyle() == "--"
        assert len(ax.patches) == 1 and ax.patches[0].get_alpha() == 0.08
    plt.close(fig)
    combined = pd.read_csv(output.with_suffix(".csv"))
    for dataset in frame.dataset.unique():
        path = tmp_path / dataset
        fig, ax = plot_paths(
            frame.loc[frame.dataset.eq(dataset)],
            dataset=dataset,
            units="nominal_hourly_dollars",
            output=path,
        )
        assert_paper_layout(fig, [ax])
        pd.testing.assert_frame_equal(
            combined.loc[combined.dataset.eq(dataset)]
            .drop(columns="dataset")
            .reset_index(drop=True),
            pd.read_csv(path.with_suffix(".csv")),
        )
        plt.close(fig)
    np.testing.assert_allclose(combined.loc[combined.method.eq("cfg"), "se"], 0.5)
    with Image.open(output.with_suffix(".png")) as png:
        np.testing.assert_allclose(png.info["dpi"], [300, 300], atol=0.01)
    pd.testing.assert_frame_equal(frame, before)


def test_cdf_panels_preserve_saved_cdfs(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [
            dict(
                dataset="cps_wage_az",
                dataset_identity="same",
                units="nominal_hourly_dollars",
                subgroup=subgroup,
                generator="smc",
                method=method,
                seed=seed,
                year=year,
                value=value + shift,
                cdf=cdf,
            )
            for subgroup, shift in [("healthcare", 0), ("cleaning", 5), ("sales", 10)]
            for method in ["real", "generated"]
            for seed in [500]
            for year in [2016, 2019]
            for value, cdf in [(0, 2 / 3), (0, 2 / 3), (2, 1)]
        ]
    )
    monkeypatch.setitem(plt.rcParams, "axes.labelsize", 17)
    caller_style = dict(plt.rcParams)
    output = tmp_path / "combined"
    fig, axes = plot_cdfs(frame, dataset="cps_wage_az", output=output, xlim=(0, 20))
    assert dict(plt.rcParams) == caller_style
    assert_paper_layout(fig, axes)
    assert [ax.get_title() for ax in axes] == ["Healthcare", "Cleaning", "Sales"]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "Observed 2016",
        "DiffLTM-SMC 2016",
        "Observed 2019",
        "DiffLTM-SMC 2019",
    ]
    assert all(ax.get_ylim() == (0, 0.3) for ax in axes)
    assert all(ax.lines[0].get_drawstyle() == "steps-post" for ax in axes)
    for ax in axes:
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0, 2 / 3, 2 / 3, 1])
    plt.close(fig)
    combined = pd.read_csv(output.with_suffix(".csv"))
    healthcare = combined.loc[combined.subgroup.eq("healthcare")]
    assert set(healthcare.value) == {0, 2}
    first = healthcare.loc[healthcare.value.eq(0)]
    np.testing.assert_allclose(first.cdf, 2 / 3)
    for subgroup in frame.subgroup.unique():
        path = tmp_path / subgroup
        fig, ax = plot_cdfs(
            frame.loc[frame.subgroup.eq(subgroup)],
            dataset="cps_wage_az",
            output=path,
            xlim=(0, 20),
        )
        assert_paper_layout(fig, [ax])
        pd.testing.assert_frame_equal(
            combined.loc[combined.subgroup.eq(subgroup)].reset_index(drop=True),
            pd.read_csv(path.with_suffix(".csv")),
        )
        plt.close(fig)
    with pytest.raises(ValueError, match="one training seed"):
        plot_cdfs(pd.concat([frame, frame.assign(seed=501)]), dataset="cps_wage_az")


def test_cdf_full_samples_ties_and_single_seed(tmp_path):
    from diffltm.economics.experiments import empirical_cdfs

    frame = empirical_cdfs(pd.Series([1.0, 1.0]), pd.Series([1.0, 2.0, 2.0, 3.0, 4.0]))
    assert frame.groupby("method").size().to_dict() == {"generated": 5, "real": 2}
    np.testing.assert_allclose(frame.loc[frame.method.eq("real"), "cdf"], [1, 1])
    np.testing.assert_allclose(
        frame.loc[frame.method.eq("generated"), "cdf"], [0.2, 0.6, 0.6, 0.8, 1]
    )
    frame = frame.assign(
        dataset="cps_wage_az",
        dataset_identity="same",
        units="nominal_hourly_dollars",
        subgroup="healthcare",
        generator="cfg",
        seed=500,
        year=2019,
    )
    output = tmp_path / "single"
    fig, ax = plot_cdfs(frame, dataset="cps_wage_az", output=output)
    values = pd.read_csv(output.with_suffix(".csv"))
    pd.testing.assert_frame_equal(values, frame, check_dtype=False)
    generated = values.loc[values.method.eq("generated")]
    np.testing.assert_allclose(generated.cdf, [0.2, 0.6, 0.6, 0.8, 1])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [0, 0.2, 0.6, 0.6, 0.8, 1])
    plt.close(fig)


def test_single_seed_paths_keep_undefined_uncertainty(tmp_path):
    frame = pd.DataFrame(
        dict(
            dataset=["cps_wage_az"] * 2,
            dataset_identity="same",
            units="nominal_hourly_dollars",
            method="cfg",
            seed=500,
            year=[2016, 2017],
            value=[8, 9],
            real=[9, 10],
            sc=[8, 9],
        )
    )
    fig, ax = plot_paths(
        frame,
        dataset="cps_wage_az",
        units="nominal_hourly_dollars",
        output=tmp_path / "single",
        figsize=(7, 4),
        xlim=(2015, 2018),
        ylim=(5, 12),
    )
    assert ax.get_xlim() == (2015, 2018) and ax.get_ylim() == (5, 12)
    np.testing.assert_allclose(fig.get_size_inches(), [7, 4])
    assert pd.read_csv(tmp_path / "single.csv").se.isna().all()
    plt.close(fig)


def test_rarity_percent_display_preserves_native_values_and_legend_order(tmp_path):
    frame = pd.DataFrame(
        [
            dict(
                dataset="fixture",
                dataset_identity="same",
                method=method,
                metric="shape",
                value=0.75,
                slice=slice_id,
                bucket=">3%",
                training_seed=seed,
                status="complete",
                omission_reason=None,
                slice_csv_sha256="same",
            )
            for method in ["cfg", "smc", "rejection", "ideal"]
            for slice_id in ["a", "b"]
            for seed in [500, 501]
        ]
    )
    fig, ax = plot_rarity(frame, dataset="fixture", output=tmp_path / "rarity")
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "Ideal",
        "DiffLTM-CFG",
        "Rejection",
        "DiffLTM-SMC",
    ]
    assert ax.yaxis.get_major_formatter()(0.75) == "75"
    assert ax.yaxis.get_major_formatter()(0.825) == "82.5"
    assert ax.yaxis.get_major_formatter()(0.875) == "87.5"
    assert ax.yaxis.get_major_formatter()(0.9995) == "99.95"
    assert ax.get_ylabel() == "Shape (%)"
    np.testing.assert_allclose(pd.read_csv(tmp_path / "rarity.csv")["mean"], 0.75)
    assert_paper_layout(fig, [ax])
    plt.close(fig)
    with pytest.raises(ValueError, match="slice identifiers"):
        plot_rarity(frame.drop(columns="slice"), dataset="fixture")
