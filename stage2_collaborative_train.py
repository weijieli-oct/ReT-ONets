"""Stage 2: physics-guided knowledge distillation.

Run this file directly in PyCharm.  The trainable reference module (RM) is
initialised from Stage 1.  A frozen teaching module (TM), previously trained
with the physical equations, supplies targets at spatio-temporal collocation
points.  The only optimisation objective is

    L_total = L_ref + beta * L_distill

where L_distill = mean(|G_ref(q, s) - G_teach(q_phys, s)|^2).
PDE residuals are deliberately NOT added to L_total here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from stage1_dataloader import create_dataloader
from stage1_model import DataDrivenNeuralOperator
from stage2_dataloader1 import create_physics_only_dataloader


# ---------------------------------------------------------------------------
# PyCharm configuration: edit these values if a different experiment is used.
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(r"E:\desktop\turbulent\Teaching")
RAW_DATA_DIR = PROJECT_DIR / "data"  # E:\desktop\turbulent\Teaching\data
OBSERVATION_DATA_DIR = PROJECT_DIR / "supervision_data"
OUTPUT_DIR = Path(r"E:\desktop\turbulent") / "collaborative_training"

STAGE1_MODEL_PATH = (
    PROJECT_DIR / "experiments" / "stage1_20260706_152425" / "models" / "best_model.pth"
)
# TM must be a physics-trained model.  The existing Stage-2 best model is used.
TEACHER_MODEL_PATH = PROJECT_DIR / "experiments" / "stage2_best" / "models" / "best_model.pth"

CONFIG: Dict[str, object] = {
    "epochs": 100,
    "batch_size": 512,
    "collocation_batch_size": 512,
    "learning_rate": 1.0e-6,
    "weight_decay": 1.0e-4,
    "beta": 0.1,
    "grad_clip": 1.0,
    "speeds": [0.095, 0.146, 0.219, 0.292],
    "time_steps": list(range(1, 21)),
    "domain_bounds": {"x_min": -0.1, "x_max": 0.3, "y_min": -0.1, "y_max": 0.1},
    "speed_range": (0.073, 0.365),
    "time_range": (0.0, 20.0),
    "num_collocation_points": 80000,
    "cylinder_radius": 0.005,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 42,
    "num_workers": 0,
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "early_stop_patience": 15,
}


def _extract_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def _infer_layers(state: Dict[str, torch.Tensor], prefix: str) -> Tuple[int, ...]:
    """Infer an MLP layout from Sequential Linear-layer weights."""
    weights = []
    for name, value in state.items():
        if name.startswith(prefix + ".") and name.endswith(".weight") and value.ndim == 2:
            index = int(name.split(".")[1])
            weights.append((index, value))
    weights.sort(key=lambda item: item[0])
    if not weights:
        raise ValueError(f"No {prefix} network weights found in checkpoint")
    return (weights[0][1].shape[1],) + tuple(weight.shape[0] for _, weight in weights)


def load_operator(path: Path, device: torch.device) -> DataDrivenNeuralOperator:
    state = _extract_state_dict(path, device)
    branch_layers = _infer_layers(state, "branch")
    trunk_layers = _infer_layers(state, "trunk")
    output_weight = state.get("output_layer.weight")
    if output_weight is None:
        raise ValueError(f"output_layer.weight is missing from {path}")
    model = DataDrivenNeuralOperator(
        branch_layers=branch_layers,
        trunk_layers=trunk_layers,
        output_dim=output_weight.shape[0],
        device=device,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint is incompatible with the operator. Missing={missing}, unexpected={unexpected}"
        )
    return model


def prepare_reference_batch(batch, device: torch.device):
    """Convert Stage-1 loader output to (q, s, g)."""
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    if inputs.ndim != 2 or inputs.shape[1] < 4:
        raise ValueError("Reference input must contain [x, y, t, inlet_speed]")
    coordinates = inputs[:, :3]
    inlet_speed = inputs[:, 3:4]
    return inlet_speed, coordinates, targets


def prepare_collocation_batch(batch, device: torch.device):
    """Convert Stage-2 loader output to (q_phys, s)."""
    values = batch.to(device)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Collocation input must contain [x, y, t, inlet_speed]")
    return values[:, 3:4], values[:, :3]


def collaborative_loss(
    reference_model: DataDrivenNeuralOperator,
    teacher_model: DataDrivenNeuralOperator,
    reference_batch,
    collocation_batch,
    beta: float,
    device: torch.device,
):
    q_ref, s_ref, g_ref = prepare_reference_batch(reference_batch, device)
    q_phys, s_collocation = prepare_collocation_batch(collocation_batch, device)

    # L_ref: retain the observational knowledge learned in Stage 1.
    reference_prediction = reference_model(q_ref, s_ref)
    loss_ref = F.mse_loss(reference_prediction, g_ref)

    # L_distill (Eq. 12): match the frozen physics-guided TM at collocation points.
    with torch.no_grad():
        teacher_prediction = teacher_model(q_phys, s_collocation)
    student_prediction = reference_model(q_phys, s_collocation)
    loss_distill = F.mse_loss(student_prediction, teacher_prediction)

    total_loss = loss_ref + beta * loss_distill
    return total_loss, loss_ref, loss_distill


def repeat_loader(loader):
    """Repeat a DataLoader without caching all batches in memory."""
    while True:
        yield from loader


def run_epoch(
    reference_model,
    teacher_model,
    reference_loader,
    collocation_loader,
    beta,
    device,
    optimizer=None,
):
    training = optimizer is not None
    reference_model.train(training)
    teacher_model.eval()
    totals = {"total": 0.0, "reference": 0.0, "distill": 0.0}
    count = 0

    # One observational batch and one collocation batch are sampled per step.
    steps = max(len(reference_loader), len(collocation_loader))
    reference_batches: Iterable = repeat_loader(reference_loader)
    collocation_batches: Iterable = repeat_loader(collocation_loader)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for _ in range(steps):
            if training:
                optimizer.zero_grad(set_to_none=True)
            total, loss_ref, loss_distill = collaborative_loss(
                reference_model,
                teacher_model,
                next(reference_batches),
                next(collocation_batches),
                beta,
                device,
            )
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(reference_model.parameters(), float(CONFIG["grad_clip"]))
                optimizer.step()
            totals["total"] += total.item()
            totals["reference"] += loss_ref.item()
            totals["distill"] += loss_distill.item()
            count += 1
    return {name: value / max(count, 1) for name, value in totals.items()}


def create_loaders(device: torch.device):
    train_ref, val_ref, _, normalizer = create_dataloader(
        batch_size=int(CONFIG["batch_size"]),
        shuffle=True,
        num_workers=int(CONFIG["num_workers"]),
        data_dir=str(OBSERVATION_DATA_DIR),
        speeds=CONFIG["speeds"],
        time_steps=CONFIG["time_steps"],
        device="cpu",
        normalize=True,
        add_inlet_speed=True,
        train_ratio=float(CONFIG["train_ratio"]),
        val_ratio=float(CONFIG["val_ratio"]),
        test_ratio=float(CONFIG["test_ratio"]),
        seed=int(CONFIG["seed"]),
    )
    train_col = create_physics_only_dataloader(
        batch_size=int(CONFIG["collocation_batch_size"]),
        shuffle=True,
        num_workers=int(CONFIG["num_workers"]),
        domain_bounds=CONFIG["domain_bounds"],
        speeds_range=CONFIG["speed_range"],
        time_range=CONFIG["time_range"],
        num_points=int(CONFIG["num_collocation_points"]),
        device="cpu",
        normalizer=normalizer,
        cylinder_radius=float(CONFIG["cylinder_radius"]),
        seed=int(CONFIG["seed"]),
    )
    val_col = create_physics_only_dataloader(
        batch_size=int(CONFIG["collocation_batch_size"]),
        shuffle=False,
        num_workers=int(CONFIG["num_workers"]),
        domain_bounds=CONFIG["domain_bounds"],
        speeds_range=CONFIG["speed_range"],
        time_range=CONFIG["time_range"],
        num_points=max(int(CONFIG["num_collocation_points"]) // 10, 1),
        device="cpu",
        normalizer=normalizer,
        cylinder_radius=float(CONFIG["cylinder_radius"]),
        seed=int(CONFIG["seed"]) + 1,
    )
    return train_ref, val_ref, train_col, val_col, normalizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Configured raw-data directory does not exist: {RAW_DATA_DIR}")

    # theta starts exactly from the Stage-1 optimum theta*.
    reference_model = load_operator(STAGE1_MODEL_PATH, device)
    teacher_model = load_operator(TEACHER_MODEL_PATH, device)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    train_ref, val_ref, train_col, val_col, normalizer = create_loaders(device)
    optimizer = Adam(
        reference_model.parameters(),
        lr=float(CONFIG["learning_rate"]),
        weight_decay=float(CONFIG["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(CONFIG["scheduler_factor"]),
        patience=int(CONFIG["scheduler_patience"]),
    )

    best_loss = float("inf")
    patience = 0
    history = []
    beta = float(CONFIG["beta"])
    for epoch in range(1, int(CONFIG["epochs"]) + 1):
        train_stats = run_epoch(
            reference_model, teacher_model, train_ref, train_col, beta, device, optimizer
        )
        val_stats = run_epoch(
            reference_model, teacher_model, val_ref, val_col, beta, device, optimizer=None
        )
        scheduler.step(val_stats["total"])
        record = {"epoch": epoch, "train": train_stats, "validation": val_stats}
        history.append(record)
        print(
            f"Epoch {epoch:03d} | "
            f"train={train_stats['total']:.6e} "
            f"(L_ref={train_stats['reference']:.6e}, L_distill={train_stats['distill']:.6e}) | "
            f"val={val_stats['total']:.6e}"
        )

        if val_stats["total"] < best_loss:
            best_loss = val_stats["total"]
            patience = 0
            torch.save(
                {
                    "model_state_dict": reference_model.state_dict(),
                    "stage1_model_path": str(STAGE1_MODEL_PATH),
                    "teacher_model_path": str(TEACHER_MODEL_PATH),
                    "beta": beta,
                    "epoch": epoch,
                    "validation_loss": best_loss,
                },
                OUTPUT_DIR / "best_model.pth",
            )
        else:
            patience += 1
            if patience >= int(CONFIG["early_stop_patience"]):
                print("Early stopping.")
                break

    torch.save(
        {"model_state_dict": reference_model.state_dict(), "beta": beta, "history": history},
        OUTPUT_DIR / "final_model.pth",
    )
    if normalizer is not None and hasattr(normalizer, "save_stats"):
        normalizer.save_stats(OUTPUT_DIR / "normalizer.pth")
    (OUTPUT_DIR / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Training finished. Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
