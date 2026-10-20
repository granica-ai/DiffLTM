"""Plot explicit saved results listed in an inventory, without run-name discovery."""

import argparse
import json
from pathlib import Path

import pandas as pd

from diffltm import plotting
from diffltm.artifacts import file_identity, identity, verify_result_training, write_json
from diffltm.data.core import dataset_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="outputs/figures")
    parser.add_argument("--only", nargs="+", help="Inventory output names to render")
    args = parser.parse_args()
    inventory = pd.read_csv(args.manifest).fillna("")
    if args.only:
        if set(args.only) - set(inventory.output):
            raise ValueError("Unknown inventory output selection")
        inventory = inventory.loc[inventory.output.isin(args.only)]
    for row in inventory.to_dict("records"):
        if row.get("status") == "deferred":
            continue
        if not row.get("results"):
            raise ValueError(f"Inventory output {row['output']} has no completed result paths")
        paths = [Path(p) for p in row["results"].split(";")]
        frames = []
        content_ids = {}
        for p in paths:
            meta = json.loads(p.with_name("manifest.json").read_text())
            if meta["status"] != "complete":
                raise ValueError(f"Incomplete input: {p}")
            verify_result_training(meta)
            expected = meta.get("files", {}).get(p.name)
            if expected is None and p.name == "summary.csv":
                expected = meta.get("summary_sha256")
            if expected is None or file_identity(p) != expected:
                raise ValueError(f"Missing or mismatched saved-result checksum: {p}")
            data = pd.read_csv(p)
            tables = [meta["table_manifest"]] if "table_manifest" in meta else []
            if "slice_manifest" in meta:
                tables.append(meta["slice_manifest"]["dataset"])
            for record in meta.get("training_runs", []):
                run = Path(record["run_dir"])
                if (run / "data.json").exists():
                    tables.append(json.loads((run / "data.json").read_text())["table"])
            for table in tables:
                if "content_hashes" in table:
                    file_id = identity({k: table[k] for k in ("name", "schema", "files")})
                    content_ids[file_id] = dataset_identity(table)
                mask = data.dataset.eq(table["name"]) & data.dataset_identity.isin(
                    [table["identity"], dataset_identity(table)]
                )
                data.loc[mask, "dataset_identity"] = dataset_identity(table)
            if "dataset_version" in data:
                data["dataset_version"] = data.dataset_version.replace({"corrected-v1": "v1"})
            frames.append(data)
        frame = pd.concat(frames, ignore_index=True)
        if "dataset_identity" in frame:
            frame["dataset_identity"] = frame.dataset_identity.replace(content_ids)
        if row.get("selection"):
            selection = json.loads(row["selection"])
            for col, value in selection.items():
                frame = frame.loc[
                    frame[col].isin(value) if isinstance(value, list) else frame[col].eq(value)
                ]
        seed_column = "training_seed" if "training_seed" in frame else "seed"
        expected_seeds = set(json.loads(row["seeds"]))
        # Figure 4 supports partial sets of training seeds and periods.
        if (
            row["plotting_function"] != "plot_reparameterization"
            and set(frame[seed_column]) != expected_seeds
        ):
            raise ValueError("Publication input does not contain the inventory's training seeds")
        if row["plotting_function"] == "format_results_table":
            groups = ["method"] + (["model"] if "model" in frame else [])
            if any(set(g[seed_column]) != expected_seeds for _, g in frame.groupby(groups)):
                raise ValueError("Each model/method requires the inventory's training seeds")
        if row["plotting_function"] == "plot_rarity":
            if set(frame.method) != set(row["sampler"].split(";")):
                raise ValueError("Publication input is missing an expected conditional method")
            if "slice_csv_sha256" not in frame:
                raise ValueError("Conditional results must use a shared frozen battery")
        output = Path(args.output) / row["output"]
        if row["plotting_function"] in {"format_results_table", "format_cps_comparisons"}:
            output.parent.mkdir(parents=True, exist_ok=True)
            getattr(plotting, row["plotting_function"])(frame).to_csv(
                output.with_suffix(".csv"), index=False
            )
        else:
            fn = getattr(plotting, row["plotting_function"])
            kwargs = dict(dataset=row["dataset"], output=output)
            if row["plotting_function"] == "plot_reparameterization":
                del kwargs["dataset"]
            for option in ["xlim", "ylim", "figsize"]:
                if row.get(option):
                    kwargs[option] = json.loads(row[option])
            if row["plotting_function"] == "plot_rarity":
                kwargs["metric"] = row["metric"]
            if row["plotting_function"] == "plot_paths":
                kwargs["units"] = row["units"]
            fig, _ = fn(frame, **kwargs)
            plotting.plt.close(fig)
        write_json(
            output.with_suffix(".inputs.json"),
            dict(
                inputs=[
                    dict(
                        path=str(p),
                        sha256=file_identity(p),
                        manifest_sha256=file_identity(p.with_name("manifest.json")),
                    )
                    for p in paths
                ],
                inventory_row=row,
            ),
        )


if __name__ == "__main__":
    main()
