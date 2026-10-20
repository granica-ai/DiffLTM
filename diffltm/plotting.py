"""Figures consume saved values with the aggregation prescribed for each experiment."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .artifacts import atomic_write, write_json

BUCKETS = [">3%", "(1%,3%]", "(0.3%,1%]", "(0.1%,0.3%]", "(0,0.1%]"]
PALETTE = {
    "cfg": "#0072B2",
    "rejection": "#C62828",
    "smc": "#2A9D8F",
    "harpoon": "#7651A8",
    "ideal": "black",
    "real": "black",
    "oracle": "black",
    "sc": "#C62828",
}
LABELS = {
    "cfg": "DiffLTM-CFG",
    "smc": "DiffLTM-SMC",
    "rejection": "Rejection",
    "harpoon": "Harpoon",
    "ideal": "Ideal",
    "real": "Observed",
    "oracle": "Oracle median",
    "sc": "Synthetic control",
}
MARKERS = {"cfg": "o", "smc": "^", "rejection": "s", "harpoon": "D", "ideal": "None"}
METHOD_ORDER = {
    method: i
    for i, method in enumerate(
        ["ideal", "real", "oracle", "sc", "cfg", "rejection", "smc", "harpoon"]
    )
}
METRIC_LABELS = {
    "shape": "Shape",
    "trend": "Trend",
    "cond_shape_all": "Conditional Shape",
    "cond_shape_rare": "Conditional Shape (rare)",
    "coefficient_error": "Squared coefficient error",
}
PANEL_LABELS = {
    "cps_wage_az": "AZ",
    "cps_wage_pa": "PA",
    "high_school": "High school",
    "bachelors": "Bachelor's",
    "healthcare": "Healthcare",
    "cleaning": "Cleaning",
    "sales": "Sales",
}
HARPOON_LABELS = {
    "onehot_representation_mse": "Harpoon (OH/Repr/MSE)",
    "learnt_logits_ce": "Harpoon (Learnt/Logits/CE)",
}
PAPER_RC = {
    "figure.dpi": 120,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlepad": 7,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.25,
    "lines.markersize": 6,
    "lines.markeredgecolor": "white",
    "lines.markeredgewidth": 0.6,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
}


def paper_axes(ncols=1, *, figsize=None, sharey=False):
    """The manuscript's physical canvas sizes, with explicit caller overrides."""
    width = 5 if ncols == 1 else 8 + 2 * (ncols - 2)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=figsize if figsize is not None else (width, 3.2),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )
    return fig, axes.ravel()


def paper_layout(
    fig, handles=None, *, xlabel=None, ylabel=None, legend_pad=4, legend_columnspacing=1.4
):
    """Shared axes and measured legend layout; wrap legends on narrow canvases."""
    for ax in fig.axes:
        ax.tick_params(length=3, width=0.6, pad=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.35)
    if xlabel:
        fig.supxlabel(xlabel, fontsize=PAPER_RC["axes.labelsize"])
    if ylabel:
        fig.supylabel(ylabel, fontsize=PAPER_RC["axes.labelsize"], multialignment="center")
    if handles is None:
        unique = {}
        for ax in fig.axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                unique.setdefault(label, handle)
        handles = list(unique.values())
    top = 1
    for ncols in range(len(handles), 0, -1):
        # Balance wrapped rows (e.g. four entries use two columns, not three).
        if ncols != int(np.ceil(len(handles) / np.ceil(len(handles) / ncols))):
            continue
        legend = fig.legend(
            handles,
            [handle.get_label() for handle in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 1),
            ncol=ncols,
            frameon=False,
            handlelength=1.8,
            handletextpad=0.4,
            columnspacing=legend_columnspacing,
        )
        fig.canvas.draw()
        bounds = legend.get_window_extent().transformed(fig.transFigure.inverted())
        if bounds.width <= 0.98 or ncols == 1:
            top = bounds.y0
            break
        legend.remove()
    fig.tight_layout(rect=(0, 0, 1, top - legend_pad / (72 * fig.get_figheight())))


def paper_errorbar(ax, x, mean, se, **kwargs):
    return ax.errorbar(x, mean, yerr=se, elinewidth=0.7, capthick=0.7, capsize=1.5, **kwargs)


def held_out_span(ax, start, stop, *, boundary=False):
    ax.axvspan(start, stop, color="#eda100", alpha=0.08, zorder=0)
    if boundary:
        ax.axvline(start, color="#7f8c8d", linestyle=":", linewidth=1)


def aggregate_runs(frame, value, groups, seed="training_seed"):
    """Mean slices within seed first, then mean/SE over run means; retain missing bins."""
    if "status" in frame and not frame.status.isin(["complete", "undefined"]).all():
        raise ValueError("Incomplete results cannot enter aggregation")
    means = (
        frame.groupby(groups + [seed], dropna=False, observed=True)[value]
        .agg(["mean", "count"])
        .reset_index()
    )
    summary = (
        means.groupby(groups, dropna=False, observed=True)
        .agg(
            mean=("mean", "mean"),
            std=("mean", "std"),
            n_runs=("mean", "count"),
            total_runs=("mean", "size"),
            n_cells=("count", "sum"),
        )
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary.n_runs)
    return summary


def aggregate_conditions(frame):
    """Seed means per condition, then equal-weight condition mean/SE."""
    if not frame.status.isin(["complete", "undefined"]).all():
        raise ValueError("Incomplete results cannot enter aggregation")
    frame = frame.copy()
    frame["harpoon_model"] = ""
    harpoon = frame.method.eq("harpoon")
    if harpoon.any():
        if "model" not in frame or not frame.loc[harpoon, "model"].isin(HARPOON_LABELS).all():
            raise ValueError("Harpoon results require a supported model identity")
        frame.loc[harpoon, "harpoon_model"] = frame.loc[harpoon, "model"]
    series = ["method", "harpoon_model"]
    if frame.duplicated(["training_seed", *series, "slice"]).any():
        raise ValueError("Duplicate run/method/slice scores")
    if frame.groupby(series).training_seed.nunique().ne(frame.training_seed.nunique()).any():
        raise ValueError("Every conditional method must include every training seed")
    sets = [set(g["slice"]) for _, g in frame.groupby([*series, "training_seed"])]
    if any(s != sets[0] for s in sets[1:]):
        raise ValueError("All methods and seeds must use identical shared slices")
    if frame.groupby("slice", observed=True).bucket.nunique().gt(1).any():
        raise ValueError("Shared slices must retain their reference buckets")
    if "slice_csv_sha256" in frame and frame.slice_csv_sha256.nunique() != 1:
        raise ValueError("Results mix different shared batteries")
    conditions = (
        frame.groupby([*series, "bucket", "slice"], observed=True)
        .agg(
            mean=("value", "mean"),
            valid_seeds=("value", "count"),
            total_seeds=("value", "size"),
            omission_reasons=("omission_reason", lambda v: "; ".join(sorted(set(v.dropna())))),
        )
        .reset_index()
    )
    summary = (
        conditions.groupby([*series, "bucket"], observed=True)
        .agg(
            mean=("mean", "mean"),
            std=("mean", "std"),
            valid_conditions=("mean", "count"),
            total_conditions=("mean", "size"),
            omission_reasons=("omission_reasons", lambda v: "; ".join(sorted(set(v) - {""}))),
        )
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary.valid_conditions)
    return conditions, summary


def check_identity(frame, dataset, units=None):
    if set(frame.dataset) != {dataset}:
        raise ValueError("Figure dataset identity mismatch")
    if "dataset_identity" in frame and frame.dataset_identity.nunique() != 1:
        raise ValueError("Figure mixes dataset content identities")
    if units is not None and ("units" not in frame or set(frame.units) != {units}):
        raise ValueError("Figure unit mismatch")


def export(fig, values, path, identities, dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", dpi=dpi)
    atomic_write(path.with_suffix(".csv"), lambda p: values.to_csv(p, index=False))
    write_json(path.with_suffix(".json"), identities)


@plt.rc_context(PAPER_RC)
def plot_rarity(
    frame,
    *,
    dataset,
    metric="shape",
    units="fraction",
    output=None,
    figsize=None,
    xlim=None,
    ylim=None,
):
    check_identity(frame, dataset)
    frame = frame.loc[frame.metric.eq(metric)].copy()
    if "task" in frame and frame.task.nunique() != 1:
        raise ValueError("Select one condition task per rarity panel")
    if frame.empty:
        raise ValueError("Missing metric")
    frame["bucket"] = pd.Categorical(frame.bucket, categories=BUCKETS, ordered=True)
    if "slice" not in frame:
        raise ValueError("Rarity plots require per-condition scores with slice identifiers")
    conditions, summary = aggregate_conditions(frame)
    fig, (ax,) = paper_axes(figsize=figsize)
    series = sorted(
        summary.groupby(["method", "harpoon_model"], sort=False),
        key=lambda pair: METHOD_ORDER.get(pair[0][0], len(METHOD_ORDER)),
    )
    for (method, variant), g in series:
        label = (
            HARPOON_LABELS[variant]
            if method == "harpoon" and variant
            else LABELS.get(method, method)
        )
        color = "#e377c2" if variant == "learnt_logits_ce" else PALETTE.get(method)
        g = g.set_index("bucket").reindex(BUCKETS)
        paper_errorbar(
            ax,
            range(5),
            g["mean"],
            g.se,
            label=label,
            color=color,
            marker=MARKERS.get(method, "o"),
            linestyle="--" if method == "ideal" else "-",
        )
    ax.set_xticks(range(5), [">3%", "1–3%", "0.3–1%", "0.1–0.3%", "≤0.1%"])
    ax.set_xlabel("Condition rarity")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    if units == "fraction":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}"))
        ax.set_ylabel(ax.get_ylabel() + " (%)")
    if metric == "coefficient_error":
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2), useMathText=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    paper_layout(fig)
    if output:
        atomic_write(
            Path(output).with_suffix(".conditions.csv"),
            lambda p: conditions.to_csv(p, index=False),
        )
        export(
            fig,
            summary,
            output,
            dict(
                dataset=dataset,
                metric=metric,
                units=units,
                dataset_identity=frame.dataset_identity.iloc[0],
            ),
        )
    return fig, ax


@plt.rc_context(PAPER_RC)
def plot_paths(frame, *, dataset, units, output=None, figsize=None, xlim=None, ylim=None):
    """Saved CPS wage paths; one panel per state holdout dataset."""
    datasets = dataset.split(";")
    if not set(datasets) <= {"cps_wage_az", "cps_wage_pa"} or units != "nominal_hourly_dollars":
        raise ValueError("Wage paths require CPS summaries in nominal hourly dollars")
    if set(frame.dataset) != set(datasets):
        raise ValueError("Path panels do not match the requested datasets")
    fig, axes = paper_axes(len(datasets), figsize=figsize, sharey=True)
    values = []
    for ax, name in zip(axes, datasets):
        data = frame.loc[frame.dataset.eq(name)]
        check_identity(data, name, units)
        parts = [data]
        for method in ["sc", "real"]:
            reference = data.assign(method=method, value=data[method])
            parts.append(reference.drop_duplicates(["seed", "year", "method"]))
        summary = aggregate_runs(pd.concat(parts), "value", ["method", "year"], seed="seed")
        order = sorted(
            summary.method.unique(), key=lambda m: METHOD_ORDER.get(m, len(METHOD_ORDER))
        )
        for method in order:
            g = summary.loc[summary.method.eq(method)].sort_values("year")
            color = PALETTE.get(method)
            ax.plot(
                g.year,
                g["mean"],
                label=LABELS.get(method, method),
                color=color,
                marker="D" if method == "sc" else MARKERS.get(method, "o"),
                linestyle="--" if method == "sc" else "-",
            )
            if method not in {"real", "sc"}:
                ax.fill_between(
                    g.year, g["mean"] - g.se, g["mean"] + g.se, color=color, alpha=0.16, linewidth=0
                )
        ax.set_title(PANEL_LABELS.get(name, name))
        ax.set_xlabel("Year")
        held_out_span(ax, 2016.5, 2019.4, boundary=True)
        ax.set_xticks(range(2010, 2020, 2))
        ax.set_xlim(xlim if xlim is not None else (2009.7, 2019.4))
        ax.set_ylim(ylim)
        values.append(summary.assign(dataset=name) if len(datasets) > 1 else summary)
    ylabel = "10th percentile wage\n($/hour)"
    if len(axes) == 1:
        axes[0].set_ylabel(ylabel)
    paper_layout(fig, ylabel=ylabel if len(axes) > 1 else None)
    if output:
        export(
            fig, pd.concat(values, ignore_index=True), output, dict(dataset=dataset, units=units)
        )
    return fig, axes[0] if len(axes) == 1 else axes


@plt.rc_context(PAPER_RC)
def plot_reparameterization(frame, *, output=None, figsize=None, xlim=None, ylim=None):
    """Figure 4: native-space CFG median estimates and fixed SC/oracle references."""
    panels = [
        ("sc_reparam_level", "Y", "Y", "Y"),
        ("sc_reparam_log", "LOG_Y", "log(Y)", "logY"),
        ("sc_reparam_quadratic", "QUAD_Y", "Y*(M-Y)", "Z"),
    ]
    if set(frame.dataset) != {p[0] for p in panels} or set(frame.method) != {"cfg"}:
        raise ValueError("Figure 4 requires the three native outcome summaries for CFG")
    if frame.M.nunique(dropna=False) != 1 or not np.isfinite(frame.M).all():
        raise ValueError("Figure 4 requires the shared retained-panel M")
    fig, axes = paper_axes(3, figsize=figsize)
    values, identities = [], []
    for ax, (dataset, units, label, display_units) in zip(axes, panels):
        data = frame.loc[frame.dataset.eq(dataset)]
        check_identity(data, dataset, units)
        if (
            set(data.oracle_kind) != {"analytic"}
            or not np.isfinite(data[["median", "sc", "oracle"]].to_numpy()).all()
        ):
            raise ValueError("Figure 4 requires finite medians and analytic oracle references")
        if data.duplicated(["seed", "period"]).any():
            raise ValueError("Duplicate reparameterization run/cell")
        if data.groupby("period")[["sc", "oracle"]].nunique().gt(1).any().any():
            raise ValueError("SC and oracle must be fixed references across training runs")
        summary = aggregate_runs(data, "median", ["method", "period"], seed="seed")
        summary = summary.sort_values("period")
        cfg = paper_errorbar(
            ax,
            summary.period,
            summary["mean"],
            se=summary.se,
            marker="o",
            linestyle="-",
            label=LABELS["cfg"],
            color=PALETTE["cfg"],
        )
        references = data[["period", "sc", "oracle"]].drop_duplicates().sort_values("period")
        parts = [summary]
        handles = {}
        for method, style in [("sc", "--"), ("oracle", "-")]:
            (handles[method],) = ax.plot(
                references.period,
                references[method],
                label=LABELS[method],
                color=PALETTE[method],
                linestyle=style,
                marker="o" if method == "oracle" else "None",
                zorder=1.9 if method == "oracle" else 2,
            )
            parts.append(
                pd.DataFrame(
                    dict(
                        period=references.period, method=method, mean=references[method], se=np.nan
                    )
                )
            )
        held_out_span(ax, 7.5, 10.5)
        ax.set_xticks(range(0, 11, 2))
        ax.set(
            title=label,
            xlabel="Period T",
            ylabel=f"Median ({display_units})",
            xlim=xlim or (-0.5, 10.5),
        )
        if ylim is not None:
            ax.set_ylim(ylim)
        panel_identity = dict(
            dataset=dataset,
            dataset_identity=data.dataset_identity.iloc[0],
            units=units,
            M=float(data.M.iloc[0]),
            oracle_kind="analytic",
        )
        values.append(pd.concat(parts, ignore_index=True).assign(**panel_identity))
        identities.append(panel_identity)
    paper_layout(fig, [handles["oracle"], handles["sc"], cfg])
    if output:
        export(
            fig,
            pd.concat(values, ignore_index=True),
            output,
            dict(
                panels=identities,
                held_out_periods=[8, 9, 10],
                statistic="native-space median; CFG mean and sample SE across training runs",
            ),
        )
    return fig, axes


@plt.rc_context(PAPER_RC)
def plot_profiles(frame, *, dataset, output=None, figsize=None, xlim=None, ylim=None):
    check_identity(frame, dataset, "log_2018_annual_wage")
    frame = frame.loc[frame.method.isin(["real", "cfg", "smc"]) & frame.age_mid.lt(50)]
    methods = [("real", "Real"), ("cfg", LABELS["cfg"]), ("smc", LABELS["smc"])]
    education = [("High school", "#0072B2"), ("Bachelor's", "#D55E00"), ("Master's", "#009E73")]
    expected = {(method, cell) for method, _ in methods for cell, _ in education}
    if set(zip(frame.method, frame.cell)) != expected:
        raise ValueError("Profiles require Real, CFG and SMC for all three education levels")
    summary = aggregate_runs(frame, "median", ["method", "cell", "age_mid"], seed="seed")
    fig, axes = paper_axes(3, figsize=figsize, sharey=True)
    for ax, (method, title) in zip(axes, methods):
        panel = summary.loc[summary.method.eq(method)]
        for cell, color in education:
            curve = panel.loc[panel.cell.eq(cell)].sort_values("age_mid")
            paper_errorbar(
                ax, curve.age_mid, curve["mean"], curve.se, label=cell, color=color, marker="o"
            )
        ax.set_title(title)
        ax.set_xlabel("Age")
        ax.set_xlim(xlim if xlim is not None else (20, 50))
        ax.set_ylim(ylim)
    axes[0].set_ylabel("Median log wage")
    paper_layout(fig)
    if output:
        export(fig, summary, output, dict(dataset=dataset, units="log_2018_annual_wage"))
    return fig, axes


def plot_coefficient_error(frame, *, dataset, output=None, figsize=None, xlim=None, ylim=None):
    check_identity(frame, dataset, "coefficient_squared_error")
    data = (
        frame.loc[frame.cell_type.eq("rarity")]
        .rename(columns={"cell": "slice", "coefficient_error": "value", "seed": "training_seed"})
        .assign(metric="coefficient_error")
    )
    return plot_rarity(
        data,
        dataset=dataset,
        metric="coefficient_error",
        units="coefficient_squared_error",
        output=output,
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
    )


@plt.rc_context(PAPER_RC)
def plot_cdfs(frame, *, dataset, output=None, figsize=None, xlim=None, ylim=None):
    """Plot the saved empirical CDFs for one CPS training seed."""
    check_identity(frame, dataset, "nominal_hourly_dollars")
    if "generator" in frame and frame.generator.nunique() != 1:
        raise ValueError("Select one generator for CDF comparisons")
    if frame.empty:
        raise ValueError("Missing CDF samples")
    if frame.seed.nunique() != 1:
        raise ValueError("Select one training seed for CPS CDF plots")
    subgroups = [s for s in PANEL_LABELS if s in set(frame.subgroup)]
    subgroups += sorted(set(frame.subgroup) - set(subgroups))
    fig, axes = paper_axes(len(subgroups), figsize=figsize, sharey=True)
    upper = []
    styles = {
        year: style for year, style in zip(sorted(frame.year.unique()), ["-", "--", ":", "-."])
    }
    generator = frame.generator.iloc[0] if "generator" in frame else "cfg"
    for ax, subgroup in zip(axes, subgroups):
        data = frame.loc[frame.subgroup.eq(subgroup)]
        for year in sorted(data.year.unique()):
            for method in ["real", "generated"]:
                g = data.loc[data.method.eq(method) & data.year.eq(year)].sort_values("value")
                display = generator if method == "generated" else method
                ax.step(
                    # Show zero probability to the left of the first observation.
                    np.r_[min(xlim[0] if xlim is not None else 0, g.value.iloc[0]), g.value],
                    np.r_[0, g.cdf],
                    where="post",
                    label=f"{LABELS.get(display, display)} {year}",
                    color=PALETTE.get(display),
                    linestyle=styles[year],
                )
        ax.axvline(11, color="0.5", linestyle=":", linewidth=1)
        ax.set(
            title=PANEL_LABELS.get(subgroup, subgroup), ylim=ylim if ylim is not None else (0, 0.3)
        )
        upper.append(data.groupby(["method", "seed", "year"]).value.quantile(0.3).max() * 1.03)
    for ax in axes:
        ax.set_xlim(xlim if xlim is not None else (0, max(upper)))
    xlabel = "Nominal hourly wage ($)"
    if len(axes) == 1:
        axes[0].set(xlabel=xlabel, ylabel="CDF")
    paper_layout(
        fig,
        xlabel=xlabel if len(axes) > 1 else None,
        ylabel="CDF" if len(axes) > 1 else None,
        legend_columnspacing=0.8,
    )
    if output:
        export(
            fig,
            frame,
            output,
            dict(dataset=dataset, units="nominal_hourly_dollars"),
        )
    return fig, axes[0] if len(axes) == 1 else axes


def format_results_table(frame):
    keys = [key for key in ("dataset", "method", "metric", "model", "slice_name", "reference_train_fraction") if key in frame]
    result = aggregate_runs(frame, "value", keys)
    return result


def format_cps_comparisons(frame):
    metrics = [
        "ks_2016",
        "ks_2019",
        "delta_ks",
        "real_sub11_2019",
        "generated_sub11_2019",
        "delta_sub11",
    ]
    long = frame.melt(
        id_vars=["dataset", "method", "subgroup", "seed"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    return aggregate_runs(long, "value", ["dataset", "method", "subgroup", "metric"], seed="seed")
