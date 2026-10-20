"""Exercise the public examples without training or downloading data."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "script,dataset,experiment,task,extra",
    [
        ("tables", "acslabour", "unconditional", "unconditional", {}),
        ("conditional", "acslabour", "conditional", "conditional", {"method": "cfg"}),
        ("CPS_minwage", "cps_wage_az", "cps", "economics", {"family": "cps", "method": "cfg"}),
    ],
)
def test_examples_route_training_evaluation_and_postprocessing(
    tmp_path, script, dataset, experiment, task, extra
):
    stub = tmp_path / "uv"
    log = tmp_path / "calls.jsonl"
    stub.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    stub.chmod(0o755)
    root = tmp_path / "results with spaces"
    cache = tmp_path / "data with spaces"
    env = {k: v for k, v in os.environ.items() if not k.startswith("DIFFLTM_")}
    env.update(
        PATH=str(tmp_path) + os.pathsep + env["PATH"],
        CALL_LOG=str(log),
        DIFFLTM_OUTPUTS=str(root),
        DIFFLTM_DATA=str(cache),
        DIFFLTM_DEVICE="cpu",
    )
    subprocess.run(
        ["bash", str(ROOT / f"reproduce/{script}.sh")], cwd=tmp_path, env=env, check=True
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == (3 if script == "CPS_minwage" else 2)
    assert calls[0][:3] == ["run", "python", "train.py"]
    train = dict(arg.split("=", 1) for arg in calls[0][3:])
    run = root / dataset / f"{experiment}_learnt_ce" / "500"
    if script == "CPS_minwage":
        assert train.pop("model.backbone") == "mlp"
        assert train.pop("training.epochs") == "2000"
    assert train == dict(
        experiment=experiment,
        dataset=dataset,
        model="learnt_ce",
        seed="500",
        device="cpu",
        output_dir=str(run),
    )
    assert calls[1][:3] == ["run", "python", "evaluate.py"]
    evaluation = dict(arg.split("=", 1) for arg in calls[1][3:])
    assert evaluation["run_dir"] == str(run)
    assert evaluation["task"] == task and evaluation["device"] == "cpu"
    assert all(evaluation[k] == v for k, v in extra.items())
    if script == "conditional":
        assert (ROOT / evaluation["slices"]).is_file()
        assert evaluation["slices"] == "slices/acslabour/equality/slices.csv"
    if script == "CPS_minwage":
        assert calls[2] == [
            "run",
            "python",
            "scripts/analyze_econ.py",
            "cps",
            "--results",
            evaluation["output_dir"],
            "--output",
            str(root / "economics/cps_wage_az"),
            "--cache",
            str(cache),
        ]


def test_conditional_inventory_retains_both_harpoon_models():
    inventory = pd.read_csv(ROOT / "paper/experiments.csv").fillna("")
    for row in inventory.to_dict("records"):
        if row["plotting_function"] != "plot_rarity":
            continue
        task = json.loads(row["selection"])["task"]
        assert "harpoon" in row["sampler"].split(";")
        paths = [p for p in row["results"].split(";") if "harpoon" in p]
        assert len(paths) == 6
        assert all(f"/{task}_harpoon/" in p for p in paths)
        assert sum("unconditional_onehot_repr" in p for p in paths) == 3
        assert sum("unconditional_learnt_ce" in p for p in paths) == 3
