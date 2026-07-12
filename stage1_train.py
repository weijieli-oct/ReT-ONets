import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import warnings
import traceback
import pandas as pd

warnings.filterwarnings('ignore')


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from stage1_dataloader import create_dataloader, DataNormalizer
    from stage1_model import DataDrivenNeuralOperator
except ImportError as e:
    print(f"input error: {e}")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")


def setup_directories(experiment_name=None):
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"stage1_{timestamp}"

    exp_dir = Path("experiments") / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    return exp_dir


def save_config(config, exp_dir):
    config_path = exp_dir / "config.json"

    config_serializable = {}
    for key, value in config.items():
        if isinstance(value, np.ndarray):
            config_serializable[key] = value.tolist()
        elif isinstance(value, Path):
            config_serializable[key] = str(value)
        else:
            config_serializable[key] = value

    with open(config_path, 'w') as f:
        json.dump(config_serializable, f, indent=2, default=str)
    print(f"config: {config_path}")
    return config_path


def setup_data_loaders(config, exp_dir=None, normalizer=None):
    print(f"\n{'=' * 60}")
    print("dataloader")
    print(f"{'=' * 60}")

    speeds = config.get("speeds", [0.095, 0.146, 0.219, 0.292])
    time_steps = config.get("time_steps", list(range(1, 21)))

    print(f"speeds ({len(speeds)} 个): {[f'{s:.3f}' for s in speeds]}")
    print(f"time steps: {time_steps}")

    try:
        train_loader, val_loader, test_loader, normalizer = create_dataloader(
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config.get("num_workers", 0),
            data_dir=config["data_dir"],
            speeds=speeds,
            time_steps=time_steps,
            device='cpu',
            normalize=config.get("normalize_data", True),
            add_inlet_speed=config.get("add_inlet_speed", True),
            train_ratio=config.get("train_ratio", 0.8),
            val_ratio=config.get("val_ratio", 0.1),
            test_ratio=config.get("test_ratio", 0.1),
            seed=config.get("seed", 42)
        )

        print(f"\n数据加载器统计:")
        print(f"  train dataset size: {len(train_loader.dataset):,} sample")
        print(f"  train batch: {len(train_loader)}")
        print(f"  valid dataset size: {len(val_loader.dataset):,} sample")
        print(f"  valid batch: {len(val_loader)}")
        print(f"  test dataset size: {len(test_loader.dataset):,} sample")
        print(f"  test batch: {len(test_loader)}")

        return train_loader, val_loader, test_loader, normalizer

    except Exception as e:
        print(f"error: {e}")
        traceback.print_exc()
        sys.exit(1)


def setup_model(config, device, normalizer=None):
    """设置模型"""
    print(f"\n{'=' * 60}")
    print(f"{'=' * 60}")

    trunk_layers = list(config["trunk_layers"])
    trunk_layers[0] = 3

    branch_layers = list(config["branch_layers"])
    branch_layers[0] = 1

    model = DataDrivenNeuralOperator(
        branch_layers=tuple(branch_layers),
        trunk_layers=tuple(trunk_layers),
        output_dim=config["output_dim"],
        device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"model:")
    print(f"  branch: {branch_layers} (for velocity)")
    print(f"  trunk: {trunk_layers} (for x, y, t)")
    print(f"  output_dim: {config['output_dim']}")
    print(f"  total_params: {total_params:,}")
    print(f"  trainable_params: {trainable_params:,}")
    if config.get("normalize_data", True):
        print(f"  normalize data: yes")
    if config.get("add_inlet_speed", True):
        print(f"  inlet_speed for feature")

    return model


def setup_optimizer(model, config):
    lr = config.get("learning_rate", 1e-3)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.get("weight_decay", 1e-4),
        betas=(0.9, 0.999)
    )

    if config.get("use_scheduler", True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get("scheduler_factor", 0.5),
            patience=config.get("scheduler_patience", 10),
            verbose=True,
            min_lr=config.get("min_lr", 1e-6)
        )
    else:
        scheduler = None

    print(f"\noptimizer setting:")
    print(f"  optim: AdamW")
    print(f"  lr: {lr}")
    print(f"  weight_decay: {config.get('weight_decay', 1e-4)}")
    print(f"  use_scheduler: {config.get('use_scheduler', True)}")
    if scheduler:
        print(f"  scheduler_factor: factor={config.get('scheduler_factor', 0.5)}, "
              f"patience={config.get('scheduler_patience', 10)}")

    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, train_stats, val_stats,
                    exp_dir, is_best=False, filename=None, config=None):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'training_stage': model.training_stage,
        'train_stats': train_stats,
        'val_stats': val_stats,
    }

    if config is not None:
        checkpoint['config'] = config

    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"

    checkpoint_path = exp_dir / "checkpoints" / filename
    torch.save(checkpoint, checkpoint_path)
    print(f"checkpoint save: {checkpoint_path}")

    if is_best:
        best_model_path = exp_dir / "models" / "best_model.pth"
        torch.save(checkpoint, best_model_path)
        print(f"best_model: {best_model_path}")

    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint[
            'scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        train_stats = checkpoint.get('train_stats', {})
        val_stats = checkpoint.get('val_stats', {})
        stage = checkpoint.get('training_stage', 'stage1')


        model.set_training_stage(stage)

        print(f"from checkpoint load: {checkpoint_path}")
        print(f"  epoch: {epoch}, stage: {stage}")

        return model, optimizer, scheduler, epoch, train_stats, val_stats

    except Exception as e:
        print(f"checkpoint error: {e}")
        traceback.print_exc()
        return model, optimizer, scheduler, 0, {}, {}


def save_training_log_to_excel(log_rows, exp_dir, filename="training_log.xlsx"):

    if not log_rows:
        print("⚠️ not log_rows")
        return None

    df = pd.DataFrame(log_rows)

    excel_path = exp_dir / "logs" / filename

    excel_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_excel(excel_path, index=False)
        print(f"✅ df.to_excel: {excel_path}")

        csv_path = excel_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ df.to_CSV: {csv_path}")

        return df, excel_path, csv_path
    except Exception as e:
        print(f"❌ save error: {e}")
        traceback.print_exc()
        return None


def plot_training_history(train_history, val_history, exp_dir, log_rows=None):
    if not train_history or not val_history:
        print(" no train_history")
        return

    epochs = range(1, len(train_history['loss']) + 1)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # loss
    axes[0, 0].plot(epochs, train_history['loss'], label='Training loss', linewidth=2, color='blue')
    axes[0, 0].plot(epochs, val_history['loss'], label='validation loss', linewidth=2, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Total Loss')

    # supervision loss
    if 'supervised' in train_history and 'supervised' in val_history:
        axes[0, 1].plot(epochs, train_history['supervised'], label='Training supervision loss', linewidth=2,
                        color='blue')
        axes[0, 1].plot(epochs, val_history['supervised'], label='validation supervision loss', linewidth=2,
                        color='red',
                        linestyle='--')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('supervision loss')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('supervision loss')

    #  R²
    if 'metrics' in train_history and 'metrics' in val_history:
        train_r2 = train_history['metrics'].get('r2_overall', [0] * len(epochs))
        val_r2 = val_history['metrics'].get('r2_overall', [0] * len(epochs))

        axes[0, 2].plot(epochs, train_r2, label='Training R²', linewidth=2, color='blue')
        axes[0, 2].plot(epochs, val_r2, label='validation R²', linewidth=2, color='red', linestyle='--')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('R²')
        axes[0, 2].set_ylim([-1, 1])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_title('Overall R²')

    # MSE
    if 'metrics' in train_history and 'metrics' in val_history:
        train_mse = train_history['metrics'].get('mse', [0] * len(epochs))
        val_mse = val_history['metrics'].get('mse', [0] * len(epochs))

        axes[1, 0].plot(epochs, train_mse, label='Training MSE', linewidth=2, color='blue')
        axes[1, 0].plot(epochs, val_mse, label='validation MSE', linewidth=2, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('MSE')

    # MAE
    if 'metrics' in train_history and 'metrics' in val_history:
        train_mae = train_history['metrics'].get('mae', [0] * len(epochs))
        val_mae = val_history['metrics'].get('mae', [0] * len(epochs))

        axes[1, 1].plot(epochs, train_mae, label='Training MAE', linewidth=2, color='blue')
        axes[1, 1].plot(epochs, val_mae, label='validation MAE', linewidth=2, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('MAE')

    # Component R² Scores (validation)
    if 'metrics' in train_history and 'metrics' in val_history:
        val_r2_p = val_history['metrics'].get('r2_p', [0] * len(epochs))
        val_r2_u = val_history['metrics'].get('r2_u', [0] * len(epochs))
        val_r2_v = val_history['metrics'].get('r2_v', [0] * len(epochs))

        axes[1, 2].plot(epochs, val_r2_p, label='R²_p (压力)', linewidth=2, color='red')
        axes[1, 2].plot(epochs, val_r2_u, label='R²_u (U速度)', linewidth=2, color='green')
        axes[1, 2].plot(epochs, val_r2_v, label='R²_v (V速度)', linewidth=2, color='blue')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('R²')
        axes[1, 2].set_ylim([-1, 1])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_title('Component R² Scores (validation)')

    # Pressure (p) R²
    if 'metrics' in train_history and 'metrics' in val_history:
        train_r2_p = train_history['metrics'].get('r2_p', [0] * len(epochs))
        val_r2_p = val_history['metrics'].get('r2_p', [0] * len(epochs))

        axes[2, 0].plot(epochs, train_r2_p, label='Train R²_p', linewidth=2, color='blue')
        axes[2, 0].plot(epochs, val_r2_p, label='Val R²_p', linewidth=2, color='red', linestyle='--')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('R²')
        axes[2, 0].set_ylim([-1, 1])
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_title('Pressure (p) R²')

    # Velocity U R²
    if 'metrics' in train_history and 'metrics' in val_history:
        train_r2_u = train_history['metrics'].get('r2_u', [0] * len(epochs))
        val_r2_u = val_history['metrics'].get('r2_u', [0] * len(epochs))

        axes[2, 1].plot(epochs, train_r2_u, label='Train R²_u', linewidth=2, color='blue')
        axes[2, 1].plot(epochs, val_r2_u, label='Val R²_u', linewidth=2, color='red', linestyle='--')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('R²')
        axes[2, 1].set_ylim([-1, 1])
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_title('Velocity U R²')

    # Velocity V R²
    if 'metrics' in train_history and 'metrics' in val_history:
        train_r2_v = train_history['metrics'].get('r2_v', [0] * len(epochs))
        val_r2_v = val_history['metrics'].get('r2_v', [0] * len(epochs))

        axes[2, 2].plot(epochs, train_r2_v, label='Train R²_v', linewidth=2, color='blue')
        axes[2, 2].plot(epochs, val_r2_v, label='Val R²_v', linewidth=2, color='red', linestyle='--')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('R²')
        axes[2, 2].set_ylim([-1, 1])
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].set_title('Velocity V R²')

    plt.tight_layout()

    # save fig
    plot_path = exp_dir / "plots" / "training_history.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"training_history: {plot_path}")

    #
    if log_rows is not None and len(log_rows) > 0:
        plot_detailed_training_history(log_rows, exp_dir)


def plot_detailed_training_history(log_rows, exp_dir):
    try:
        df = pd.DataFrame(log_rows)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        epochs = df['epoch']

        # Training vs validation Loss
        axes[0, 0].plot(epochs, df['train_loss'], 'b-', label='Training loss', linewidth=2)
        axes[0, 0].plot(epochs, df['val_loss'], 'r-', label='validation loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Training vs validation Loss')

        # MSE
        axes[0, 1].plot(epochs, df['train_mse'], 'b-', label='Training MSE', linewidth=2)
        axes[0, 1].plot(epochs, df['val_mse'], 'r-', label='validation MSE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Training vs validation MSE')

        # MAE
        axes[1, 0].plot(epochs, df['train_mae'], 'b-', label='Training MAE', linewidth=2)
        axes[1, 0].plot(epochs, df['val_mae'], 'r-', label='validation MAE', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Training vs validation MAE')

        # R²
        axes[1, 1].plot(epochs, df['train_r2_overall'], 'b-', label='Training R²', linewidth=2)
        axes[1, 1].plot(epochs, df['val_r2_overall'], 'r-', label='validation R²', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Training vs validation R²')
        axes[1, 1].set_ylim([-1, 1])

        plt.tight_layout()

        detailed_plot_path = exp_dir / "plots" / "detailed_training_history.png"
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"detailed_plot: {detailed_plot_path}")

    except Exception as e:
        print(f"detailed_plot error: {e}")
        traceback.print_exc()


def prepare_batch_data_for_model(batch_data, device, add_inlet_speed=True):

    inputs, outputs = batch_data

    inputs = inputs.to(device)
    outputs = outputs.to(device)

    if add_inlet_speed:
        # [batch_size, 4]
        coordinates = inputs[:, 0:3]  # x, y, t
        inlet_speed = inputs[:, 3:4]  # U_in
    else:
        # 输入格式: [batch_size, 3]
        coordinates = inputs
        inlet_speed = torch.zeros((inputs.shape[0], 1), device=device)


    batch_size = inputs.shape[0]
    data_types = torch.zeros(batch_size, 1, device=device)  #
    time_steps = torch.zeros(batch_size, 1, device=device)  #

    return (inlet_speed, coordinates, outputs, data_types, time_steps)


def train_epoch(model, train_loader, optimizer, device, epoch, config):

    model.train()
    total_loss = 0.0
    supervised_loss_total = 0.0
    boundary_loss_total = 0.0
    initial_loss_total = 0.0

    epoch_metrics = {
        'mse': 0.0,
        'mae': 0.0,
        'r2_p': 0.0,
        'r2_u': 0.0,
        'r2_v': 0.0,
        'r2_overall': 0.0
    }
    supervised_batches = 0

    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - train", leave=False)

    for batch_idx, batch_data in enumerate(pbar):

        batch_data_for_model = prepare_batch_data_for_model(
            batch_data, device, config.get("add_inlet_speed", True)
        )

        optimizer.zero_grad()

        loss, loss_components, metrics = model.compute_loss(batch_data_for_model)

        loss.backward()

        if config.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        optimizer.step()

        total_loss += loss.item()
        supervised_loss_total += loss_components.get('supervised', 0.0)
        boundary_loss_total += loss_components.get('boundary', 0.0)
        initial_loss_total += loss_components.get('initial', 0.0)

        if loss_components.get('supervised', 0) > 0:
            supervised_batches += 1
            epoch_metrics['mse'] += metrics.get('mse', 0.0)
            epoch_metrics['mae'] += metrics.get('mae', 0.0)
            epoch_metrics['r2_p'] += metrics.get('r2_p', 0.0)
            epoch_metrics['r2_u'] += metrics.get('r2_u', 0.0)
            epoch_metrics['r2_v'] += metrics.get('r2_v', 0.0)
            epoch_metrics['r2_overall'] += metrics.get('r2_overall', 0.0)

        postfix_dict = {
            'Loss': f'{loss.item():.4e}',
            'Sup': f'{loss_components.get("supervised", 0):.4e}',
            'R2': f'{metrics.get("r2_overall", 0):.4f}'
        }

        pbar.set_postfix(postfix_dict)

    avg_total_loss = total_loss / num_batches
    avg_supervised_loss = supervised_loss_total / num_batches
    avg_boundary_loss = boundary_loss_total / num_batches
    avg_initial_loss = initial_loss_total / num_batches

    if supervised_batches > 0:
        for key in epoch_metrics:
            epoch_metrics[key] /= supervised_batches

    return {
        'loss': avg_total_loss,
        'supervised': avg_supervised_loss,
        'boundary': avg_boundary_loss,
        'initial': avg_initial_loss,
        'metrics': epoch_metrics
    }


def validate(model, val_loader, device, config):

    model.eval()

    val_total_loss = 0.0
    val_supervised_loss = 0.0
    val_boundary_loss = 0.0
    val_initial_loss = 0.0

    val_metrics = {
        'mse': 0.0,
        'mae': 0.0,
        'r2_p': 0.0,
        'r2_u': 0.0,
        'r2_v': 0.0,
        'r2_overall': 0.0
    }
    val_batches = 0
    supervised_val_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="valid", leave=False)
        for batch_data in pbar:

            batch_data_for_model = prepare_batch_data_for_model(
                batch_data, device, config.get("add_inlet_speed", True)
            )

            loss, loss_components, metrics = model.compute_loss(batch_data_for_model)

            val_total_loss += loss.item()
            val_supervised_loss += loss_components.get('supervised', 0.0)
            val_boundary_loss += loss_components.get('boundary', 0.0)
            val_initial_loss += loss_components.get('initial', 0.0)

            if loss_components.get('supervised', 0) > 0:
                supervised_val_batches += 1
                val_metrics['mse'] += metrics.get('mse', 0.0)
                val_metrics['mae'] += metrics.get('mae', 0.0)
                val_metrics['r2_p'] += metrics.get('r2_p', 0.0)
                val_metrics['r2_u'] += metrics.get('r2_u', 0.0)
                val_metrics['r2_v'] += metrics.get('r2_v', 0.0)
                val_metrics['r2_overall'] += metrics.get('r2_overall', 0.0)

            val_batches += 1

    if val_batches > 0:
        val_total_loss /= val_batches
        val_supervised_loss /= val_batches
        val_boundary_loss /= val_batches
        val_initial_loss /= val_batches

        if supervised_val_batches > 0:
            for key in val_metrics:
                val_metrics[key] /= supervised_val_batches

    return {
        'loss': val_total_loss,
        'supervised': val_supervised_loss,
        'boundary': val_boundary_loss,
        'initial': val_initial_loss,
        'metrics': val_metrics
    }


def test(model, test_loader, device, config):
    return validate(model, test_loader, device, config)


def train_model(config, exp_dir, checkpoint_path=None):
    print(f"\n{'=' * 60}")
    print(f"{'=' * 60}")

    training_logs = []

    normalizer = None
    if config.get("normalize_data", True):
        normalizer = DataNormalizer('zscore')
        normalizer_path = exp_dir / "models" / "normalizer.pth"
        if normalizer_path.exists():
            normalizer.load_stats(normalizer_path)


    train_loader, val_loader, test_loader, normalizer = setup_data_loaders(config, exp_dir, normalizer)

    if normalizer and config.get("normalize_data", True) and hasattr(normalizer, 'save_stats'):
        normalizer_path = exp_dir / "models" / "normalizer.pth"
        normalizer.save_stats(normalizer_path)

    model = setup_model(config, device, normalizer)

    optimizer, scheduler = setup_optimizer(model, config)

    train_history = {
        'loss': [],
        'supervised': [],
        'boundary': [],
        'initial': [],
        'metrics': {
            'mse': [], 'mae': [],
            'r2_p': [], 'r2_u': [], 'r2_v': [], 'r2_overall': []
        }
    }

    val_history = {
        'loss': [],
        'supervised': [],
        'boundary': [],
        'initial': [],
        'metrics': {
            'mse': [], 'mae': [],
            'r2_p': [], 'r2_u': [], 'r2_v': [], 'r2_overall': []
        }
    }

    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    patience_counter = 0
    patience = config.get("early_stop_patience", 20)

    num_epochs = config.get("epochs", 100)


    start_epoch = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        model, optimizer, scheduler, start_epoch, train_history, val_history = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )
        print(f" epoch {start_epoch}")


    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)


        train_stats = train_epoch(model, train_loader, optimizer, device, epoch, config)

        val_stats = validate(model, val_loader, device, config)

        train_history['loss'].append(train_stats['loss'])
        train_history['supervised'].append(train_stats['supervised'])
        train_history['boundary'].append(train_stats['boundary'])
        train_history['initial'].append(train_stats['initial'])

        for key in train_history['metrics']:
            train_history['metrics'][key].append(train_stats['metrics'].get(key, 0.0))

        val_history['loss'].append(val_stats['loss'])
        val_history['supervised'].append(val_stats['supervised'])
        val_history['boundary'].append(val_stats['boundary'])
        val_history['initial'].append(val_stats['initial'])

        for key in val_history['metrics']:
            val_history['metrics'][key].append(val_stats['metrics'].get(key, 0.0))

        current_lr = optimizer.param_groups[0]['lr']

        log_entry = {
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'train_supervised_loss': train_stats['supervised'],
            'train_boundary_loss': train_stats['boundary'],
            'train_initial_loss': train_stats['initial'],
            'train_mse': train_stats['metrics']['mse'],
            'train_mae': train_stats['metrics']['mae'],
            'train_r2_p': train_stats['metrics']['r2_p'],
            'train_r2_u': train_stats['metrics']['r2_u'],
            'train_r2_v': train_stats['metrics']['r2_v'],
            'train_r2_overall': train_stats['metrics']['r2_overall'],
            'val_loss': val_stats['loss'],
            'val_supervised_loss': val_stats['supervised'],
            'val_boundary_loss': val_stats['boundary'],
            'val_initial_loss': val_stats['initial'],
            'val_mse': val_stats['metrics']['mse'],
            'val_mae': val_stats['metrics']['mae'],
            'val_r2_p': val_stats['metrics']['r2_p'],
            'val_r2_u': val_stats['metrics']['r2_u'],
            'val_r2_v': val_stats['metrics']['r2_v'],
            'val_r2_overall': val_stats['metrics']['r2_overall'],
            'learning_rate': current_lr
        }

        training_logs.append(log_entry)

        # results
        print(f"train loss: {train_stats['loss']:.6f}")
        print(f"valid loss: {val_stats['loss']:.6f}")
        print(f"training:")
        print(f"  MSE: {train_stats['metrics']['mse']:.6f}, MAE: {train_stats['metrics']['mae']:.6f}")
        print(f"  R²_p: {train_stats['metrics']['r2_p']:.4f}, R²_u: {train_stats['metrics']['r2_u']:.4f}, "
              f"R²_v: {train_stats['metrics']['r2_v']:.4f}, R²_overall: {train_stats['metrics']['r2_overall']:.4f}")

        print(f"valid:")
        print(f"  MSE: {val_stats['metrics']['mse']:.6f}, MAE: {val_stats['metrics']['mae']:.6f}")
        print(f"  R²_p: {val_stats['metrics']['r2_p']:.4f}, R²_u: {val_stats['metrics']['r2_u']:.4f}, "
              f"R²_v: {val_stats['metrics']['r2_v']:.4f}, R²_overall: {val_stats['metrics']['r2_overall']:.4f}")

        if scheduler:
            scheduler.step(val_stats['loss'])

        is_best = val_stats['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_stats['loss']
            best_val_r2 = val_stats['metrics']['r2_overall']
            patience_counter = 0
            print(f"best_val_loss: {best_val_loss:.6f}, R²_overall: {best_val_r2:.4f}")
        else:
            patience_counter += 1

        if epoch % config.get("save_interval", 10) == 0 or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, train_stats, val_stats,
                            exp_dir, is_best, f"checkpoint_epoch_{epoch}.pth", config)

        if patience_counter >= patience:
            print(f"\n早停触发！在 {epoch} 轮后停止训练")
            break

    # test
    print(f"\n{'=' * 60}")
    print("test")
    print(f"{'=' * 60}")

    test_stats = test(model, test_loader, device, config)
    print(f"test loss: {test_stats['loss']:.6f}")
    print(f"test:")
    print(f"  MSE: {test_stats['metrics']['mse']:.6f}, MAE: {test_stats['metrics']['mae']:.6f}")
    print(f"  R²_p: {test_stats['metrics']['r2_p']:.4f}, R²_u: {test_stats['metrics']['r2_u']:.4f}, "
          f"R²_v: {test_stats['metrics']['r2_v']:.4f}, R²_overall: {test_stats['metrics']['r2_overall']:.4f}")


    test_log_entry = {
        'epoch': 'test',
        'train_loss': 0.0,
        'train_supervised_loss': 0.0,
        'train_boundary_loss': 0.0,
        'train_initial_loss': 0.0,
        'train_mse': 0.0,
        'train_mae': 0.0,
        'train_r2_p': 0.0,
        'train_r2_u': 0.0,
        'train_r2_v': 0.0,
        'train_r2_overall': 0.0,
        'val_loss': 0.0,
        'val_supervised_loss': 0.0,
        'val_boundary_loss': 0.0,
        'val_initial_loss': 0.0,
        'val_mse': 0.0,
        'val_mae': 0.0,
        'val_r2_p': 0.0,
        'val_r2_u': 0.0,
        'val_r2_v': 0.0,
        'val_r2_overall': 0.0,
        'test_loss': test_stats['loss'],
        'test_supervised_loss': test_stats['supervised'],
        'test_boundary_loss': test_stats['boundary'],
        'test_initial_loss': test_stats['initial'],
        'test_mse': test_stats['metrics']['mse'],
        'test_mae': test_stats['metrics']['mae'],
        'test_r2_p': test_stats['metrics']['r2_p'],
        'test_r2_u': test_stats['metrics']['r2_u'],
        'test_r2_v': test_stats['metrics']['r2_v'],
        'test_r2_overall': test_stats['metrics']['r2_overall'],
        'learning_rate': optimizer.param_groups[0]['lr']
    }

    training_logs.append(test_log_entry)

    # final_model
    final_model_path = exp_dir / "models" / "final_model.pth"
    model.save_model(str(final_model_path))
    print(f"\nfinal_model: {final_model_path}")

    if normalizer and hasattr(normalizer, 'save_stats'):
        normalizer_path = exp_dir / "models" / "normalizer.pth"
        normalizer.save_stats(normalizer_path)
        print(f"normalizer: {normalizer_path}")

    if training_logs:
        save_training_log_to_excel(training_logs, exp_dir, "training_log.xlsx")

        if len(training_logs) > 1:
            training_epochs_logs = [log for log in training_logs if log['epoch'] != 'test']
            if training_epochs_logs:
                save_training_log_to_excel(training_epochs_logs, exp_dir, "training_epochs_log.xlsx")

    plot_training_history(train_history, val_history, exp_dir, training_logs)

    history_path = exp_dir / "training_history.json"
    with open(history_path, 'w') as f:

        history_to_save = {
            'train_history': {
                'loss': [float(x) for x in train_history['loss']],
                'supervised': [float(x) for x in train_history.get('supervised', [])],
                'boundary': [float(x) for x in train_history.get('boundary', [])],
                'initial': [float(x) for x in train_history.get('initial', [])],
                'metrics': {}
            },
            'val_history': {
                'loss': [float(x) for x in val_history['loss']],
                'supervised': [float(x) for x in val_history.get('supervised', [])],
                'boundary': [float(x) for x in val_history.get('boundary', [])],
                'initial': [float(x) for x in val_history.get('initial', [])],
                'metrics': {}
            },
            'test_stats': {
                'loss': float(test_stats['loss']) if 'loss' in test_stats else 0.0,
                'supervised': float(test_stats.get('supervised', 0.0)),
                'boundary': float(test_stats.get('boundary', 0.0)),
                'initial': float(test_stats.get('initial', 0.0)),
                'metrics': {k: float(v) for k, v in test_stats.get('metrics', {}).items()}
            }
        }

        for key in train_history['metrics']:
            history_to_save['train_history']['metrics'][key] = [float(x) for x in train_history['metrics'][key]]

        for key in val_history['metrics']:
            history_to_save['val_history']['metrics'][key] = [float(x) for x in val_history['metrics'][key]]

        json.dump(history_to_save, f, indent=2, default=str)

    print(f"history: {history_path}")

    return model, normalizer, training_logs


def parse_args():
    parser = argparse.ArgumentParser(description='Ref model')

    parser.add_argument('--data_dir', type=str, default='supervision_data', help='Supervisory Data Directory')

    parser.add_argument('--speeds', type=float, nargs='+',
                        default=[0.095, 0.146, 0.219, 0.292], help='Training speed list')
    parser.add_argument('--time_steps', type=int, nargs='+',
                        default=list(range(1, 21)), help='Time step list')

    parser.add_argument('--branch_layers', type=int, nargs='+', default=[1, 64, 128, 128, 256, 512, 256, 128],
                        help='Branch network layer structure')
    parser.add_argument('--trunk_layers', type=int, nargs='+', default=[3, 64, 128, 128, 256, 512, 256, 128],
                        help='Trunk network layer structure')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dimension')

    parser.add_argument('--epochs', type=int, default=100, help='train epoch')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')

    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--normalize_data', action='store_true', default=True,
                        help='Should the data be standardized?')
    parser.add_argument('--no_normalize', action='store_false', dest='normalize_data',
                        help='Not standardizing the data')

    parser.add_argument('--add_inlet_speed', action='store_true', default=True,
                        help='Should the entry speed be regarded as an input feature?')
    parser.add_argument('--no_add_inlet_speed', action='store_false', dest='add_inlet_speed',
                        help='Do not use the entry speed as an input feature')

    parser.add_argument('--use_scheduler', action='store_true', default=True,
                        help='Whether to use a learning rate scheduler')
    parser.add_argument('--no_scheduler', action='store_false', dest='use_scheduler',
                        help='Do not use the learning rate scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Learning rate scheduler decay factor')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help='Learning rate scheduler patience value')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')

    parser.add_argument('--early_stop_patience', type=int, default=20, help='early stop patience')
    parser.add_argument('--save_interval', type=int, default=10, help='Retention interval')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading worker processes')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment Name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')

    parser.add_argument('--save_logs', action='store_true', default=True,
                        help='Do you want to save the training logs to Excel?')
    parser.add_argument('--no_save_logs', action='store_false', dest='save_logs',
                        help='Do not save the training logs to Excel')

    return parser.parse_args()


def main():
    """主函数"""

    args = parse_args()

    config = {

        "data_dir": args.data_dir,

        "speeds": args.speeds,
        "time_steps": args.time_steps,


        "branch_layers": args.branch_layers,
        "trunk_layers": args.trunk_layers,
        "output_dim": args.output_dim,


        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,


        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,


        "normalize_data": args.normalize_data,


        "add_inlet_speed": args.add_inlet_speed,


        "use_scheduler": args.use_scheduler,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "min_lr": args.min_lr,


        "early_stop_patience": args.early_stop_patience,
        "save_interval": args.save_interval,
        "num_workers": args.num_workers,


        "save_logs": args.save_logs,
    }


    exp_dir = setup_directories(args.experiment_name)


    save_config(config, exp_dir)


    model, normalizer, training_logs = train_model(config, exp_dir, args.checkpoint)

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"All the results are saved in: {exp_dir}")
    print(f"The model has been saved and can be used for subsequent physical-driven training.")
    if config.get("save_logs", True) and training_logs:
        print(f"The training log has been saved as an Excel file")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()