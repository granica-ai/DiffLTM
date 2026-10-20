"""Full-budget AdamW training, progressing validation RNG and denoiser-only EMA."""

import copy
import random
from pathlib import Path

import numpy as np
import torch

from .artifacts import atomic_write, file_identity, identity, request_seed, write_json
from .data.core import load_table, split_table
from .data.preprocessing import Preprocessor, imputed_frame
from .models.model import DiffLTM
from .periodic import DiagnosticJob, PeriodicWorker, sample_snapshot, write_status


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path, payload):
    atomic_write(path, lambda p: torch.save(payload, p))


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = DiffLTM(checkpoint["preprocessor"], checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def train(config):
    cfg = copy.deepcopy(config)
    seed = cfg["seed"]
    device = torch.device(cfg["device"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # Hydra creates its own directory before entering the trainer.
    if any(path.name != "hydra" for path in out.iterdir()):
        raise FileExistsError(f"Existing run: use a fresh output_dir: {out}")
    write_json(out / "config.json", cfg)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    table = load_table(cfg["dataset"]["name"], cfg["cache"])
    splits, split_id = split_table(table, seed, cfg["cache"])
    train_frame = table.frame.iloc[splits["train"]]
    val_frame = table.frame.iloc[splits["validation"]]
    p = Preprocessor(table.schema, cfg["dataset"]["numerical"], seed).fit(train_frame)
    # Downloaders may use global RNGs; cold and warm acquisition must train identically.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DiffLTM(p, cfg["model"]).to(device)
    tc = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    ema = copy.deepcopy(model.denoiser).eval() if tc["ema_decay"] else None
    if ema is not None:
        ema.requires_grad_(False)
    amp = bool(tc["amp"] and device.type == "cuda")
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    train_tensor = torch.as_tensor(p.transform(train_frame), device=device)
    val_tensor = torch.as_tensor(p.transform(val_frame), device=device)
    train_frame = imputed_frame(train_frame, p.schema, p.fill)
    val_frame = imputed_frame(val_frame, p.schema, p.fill)
    best, selected_epoch, updates = float("inf"), None, 0
    selected_state = None
    history = []
    atomic_write(out / "split.npz", lambda path: _save_splits(path, splits))
    write_json(
        out / "data.json",
        dict(
            table=table.manifest,
            split_id=split_id,
            splits_sha256=file_identity(out / "split.npz"),
            seed=seed,
            counts={k: len(v) for k, v in splits.items()},
        ),
    )
    worker = (
        PeriodicWorker(out / "periodic", p.schema, train_frame, val_frame)
        if tc["periodic_every"]
        else None
    )
    run_id = identity([table.manifest["identity"], split_id, cfg["model"], tc, seed])
    status, epoch = "running", 0

    def progress(phase):
        write_json(
            out / "status.json",
            dict(status=status, phase=phase, epoch=epoch, updates=updates, run_id=run_id),
        )

    progress("training")

    def snapshot():
        state = cpu_state(model)
        if ema is not None:
            state.update(
                {"denoiser." + k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
            )
        return state

    def payload(state):
        return dict(
            format_version=1,
            config=cfg,
            schema=table.schema,
            preprocessor=p,
            table_manifest=table.manifest,
            split_id=split_id,
            splits=splits,
            run_id=run_id,
            model=state,
            epoch=epoch,
            final_epoch=epoch,
            selected_epoch=selected_epoch,
            selected_loss=best,
            selection="minimum_ema_validation_loss"
            if ema is not None
            else "minimum_validation_loss",
            updates=updates,
            history=history,
            status=status,
        )

    try:
        for epoch in range(1, tc["epochs"] + 1):
            progress("training")
            model.train()
            total, count = 0.0, 0
            order = torch.randperm(len(train_tensor), device=device)
            for indices in order.split(tc["batch_size"]):
                if worker is not None:
                    worker.poll()
                batch = train_tensor[indices]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
                    loss = model.loss(batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                updates += 1
                if ema is not None:
                    with torch.no_grad():
                        for avg, live in zip(ema.parameters(), model.denoiser.parameters()):
                            avg.lerp_(live, 1 - tc["ema_decay"])
                total += float(loss.detach()) * len(batch)
                count += len(batch)
                if not torch.isfinite(loss):
                    status = "nonfinite_training_loss"
                    break
            if status != "running":
                break
            scheduler.step()
            row = dict(epoch=epoch, train_loss=total / count, updates=updates)
            if epoch % tc["val_every"] == 0:
                model.eval()
                live = model.denoiser
                if ema is not None:
                    model.denoiser = ema
                try:
                    with torch.no_grad():
                        losses = []
                        for batch in val_tensor.split(tc["batch_size"]):
                            with torch.autocast(
                                device_type=device.type, enabled=amp, dtype=amp_dtype
                            ):
                                losses.append(float(model.loss(batch)))
                    validation = float(
                        np.mean(losses)
                    )  # Retain batch-mean reduction, including final short batch.
                finally:
                    model.denoiser = live
                row["validation_loss"] = validation
                if validation < best:
                    best, selected_epoch, selected_state = validation, epoch, snapshot()
                    save_checkpoint(out / "selected.pt", payload(selected_state))
            history.append(row)
            print(row, flush=True)
            if worker is not None and epoch % tc["periodic_every"] == 0:
                worker.poll()
                progress("periodic_sampling")
                diagnostic_seed = request_seed(seed, epoch, "periodic")
                try:
                    synthetic, generation = sample_snapshot(
                        model, ema, len(train_frame), diagnostic_seed
                    )
                except BaseException as exc:
                    job = DiagnosticJob(
                        epoch,
                        diagnostic_seed,
                        out / "periodic" / f"epoch-{epoch}.parquet",
                        generation=dict(recovery=getattr(exc, "diagnostics", None)),
                    )
                    write_status(job, "failed", error=repr(exc))
                    raise
                worker.submit(epoch, diagnostic_seed, synthetic, tc["periodic_mle"], generation)
                del synthetic
                progress("training")
            if epoch % tc["save_every"] == 0:
                save_checkpoint(out / "final.pt", payload(snapshot()))
        if worker is not None:
            progress("waiting_for_periodic")
            closing, worker = worker, None
            closing.close()
        if status == "running":
            status = "complete" if selected_state is not None else "no_finite_validation"
    except BaseException:
        status = "failed"
        progress("failed")
        raise
    finally:
        if worker is not None:
            worker.close()
    save_checkpoint(out / "final.pt", payload(snapshot()))
    if selected_state is not None:
        save_checkpoint(out / "selected.pt", payload(selected_state))
    write_json(
        out / "status.json",
        dict(
            status=status,
            phase=status,
            epoch=epoch,
            selected_epoch=selected_epoch,
            selected_loss=best if np.isfinite(best) else None,
            updates=updates,
            run_id=run_id,
        ),
    )
    if selected_state is None:
        raise RuntimeError(
            "No finite improving validation checkpoint; inspect final.pt and configured cadence"
        )
    if status != "complete":
        raise RuntimeError(f"Training failed: {status}; inspect final.pt and status.json")
    return out / "selected.pt"


def _save_splits(path, splits):
    with path.open("wb") as f:
        np.savez_compressed(f, **splits)
