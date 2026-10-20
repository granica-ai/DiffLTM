"""Content identities and atomic, restart-safe artifact writes."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def file_identity(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    os.close(fd)
    try:
        writer(Path(tmp))
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(
        path,
        lambda p: p.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"),
    )


def request_seed(seed, request, purpose):
    return int(identity([seed, request, purpose])[:8], 16)


def completed_training(run_dir, config=None):
    """Verify full-budget completion and the requested experiment settings."""
    run = Path(run_dir)
    if not all((run / name).is_file() for name in ("status.json", "config.json", "selected.pt")):
        raise ValueError(
            f"Final evaluation requires completed training; use a fresh output_dir: {run}"
        )
    status = json.loads((run / "status.json").read_text())
    saved_config = json.loads((run / "config.json").read_text())
    if status["status"] != "complete" or status["epoch"] != saved_config["training"]["epochs"]:
        raise ValueError(
            f"Final evaluation requires completed training; use a fresh output_dir: {run}"
        )
    if config is not None and any(
        config[key] != saved_config[key] for key in ("seed", "dataset", "model", "training")
    ):
        raise ValueError(f"Training configuration mismatch; use a fresh output_dir: {run}")
    fields = ("status", "epoch", "selected_epoch", "selected_loss", "updates", "run_id")
    return dict(run_dir=str(run.resolve()), **{k: status[k] for k in fields if k in status})


def verify_result_training(manifest):
    """Validate saved completion evidence without requiring the original run location."""
    records = manifest.get("training_runs")
    if not records:
        raise ValueError("Result has no completed-training evidence; evaluate it again")
    for record in records:
        if record.get("status") != "complete" or not record.get("run_id"):
            raise ValueError("Result has no completed-training evidence")


def evaluation_output(parent, output=None):
    """Create a fresh evaluation directory; never replace historical results."""
    if output is not None:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=False)
        return path
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="evaluation-", dir=parent))


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
