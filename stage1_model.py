import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List, Union
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import os


class DataDrivenNeuralOperator(nn.Module):

    def __init__(self,
                 branch_layers: Tuple[int, ...] = (1, 128, 128, 128),
                 trunk_layers: Tuple[int, ...] = (3, 128, 128, 128),
                 output_dim: int = 3,
                 device: torch.device = None):

        super(DataDrivenNeuralOperator, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.branch = self._build_mlp(branch_layers)
        self.trunk = self._build_mlp(trunk_layers)

        hidden_dim = branch_layers[-1]

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.training_stage = 'stage1'

        self.to(self.device)



    def _build_mlp(self, layers: Tuple[int, ...]) -> nn.Module:
        mlp_layers = []
        for i in range(len(layers) - 2):
            mlp_layers.append(nn.Linear(layers[i], layers[i + 1]))
            mlp_layers.append(nn.Tanh())
        mlp_layers.append(nn.Linear(layers[-2], layers[-1]))

        return nn.Sequential(*mlp_layers)

    def forward(self, inlet_speed: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:

        inlet_speed = inlet_speed.to(self.device)
        coordinates = coordinates.to(self.device)

        branch_out = self.branch(inlet_speed)  # [batch_size, hidden_dim]

        trunk_out = self.trunk(coordinates)  # [batch_size, hidden_dim]

        multiplied = branch_out * trunk_out  # [batch_size, hidden_dim]

        outputs = self.output_layer(multiplied)  # [batch_size, 3]

        return outputs

    def set_training_stage(self, stage: str):
        if stage not in ['stage1', 'stage2']:
            raise ValueError(f"training_stage: {stage}")

        self.training_stage = stage
        print(f"training_stage: {stage}")

    def compute_loss(self, batch_data: Tuple) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:

        inlet_speeds, inputs, outputs, data_types, _ = batch_data

        inlet_speeds = inlet_speeds.to(self.device)
        inputs = inputs.to(self.device)
        outputs = outputs.to(self.device)
        data_types = data_types.to(self.device)

        supervised_mask = (data_types == 0).squeeze()

        if not supervised_mask.any():
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_components = {
                'total': 0.0,
                'supervised': 0.0
            }
            metrics = {
                'mse': 0.0,
                'mae': 0.0,
                'r2_p': 0.0,
                'r2_u': 0.0,
                'r2_v': 0.0,
                'r2_overall': 0.0
            }
            return loss, loss_components, metrics

        sup_inlet = inlet_speeds[supervised_mask]
        sup_inputs = inputs[supervised_mask]
        sup_targets = outputs[supervised_mask]

        sup_preds = self.forward(sup_inlet, sup_inputs)

        supervised_loss = F.mse_loss(sup_preds, sup_targets)
        total_loss = supervised_loss

        with torch.no_grad():
            mse = F.mse_loss(sup_preds, sup_targets).item()
            mae = F.l1_loss(sup_preds, sup_targets).item()

            sup_preds_np = sup_preds.detach().cpu().numpy()
            sup_targets_np = sup_targets.detach().cpu().numpy()

            r2_p = 0.0
            r2_u = 0.0
            r2_v = 0.0

            mask_p = np.isfinite(sup_targets_np[:, 0]) & np.isfinite(sup_preds_np[:, 0])
            if np.sum(mask_p) > 1:
                try:
                    r2_p = r2_score(sup_targets_np[mask_p, 0], sup_preds_np[mask_p, 0])
                except:
                    r2_p = 0.0

            mask_u = np.isfinite(sup_targets_np[:, 1]) & np.isfinite(sup_preds_np[:, 1])
            if np.sum(mask_u) > 1:
                try:
                    r2_u = r2_score(sup_targets_np[mask_u, 1], sup_preds_np[mask_u, 1])
                except:
                    r2_u = 0.0

            mask_v = np.isfinite(sup_targets_np[:, 2]) & np.isfinite(sup_preds_np[:, 2])
            if np.sum(mask_v) > 1:
                try:
                    r2_v = r2_score(sup_targets_np[mask_v, 2], sup_preds_np[mask_v, 2])
                except:
                    r2_v = 0.0

            all_targets = sup_targets_np.reshape(-1)
            all_preds = sup_preds_np.reshape(-1)
            mask_all = np.isfinite(all_targets) & np.isfinite(all_preds)
            if np.sum(mask_all) > 1:
                try:
                    r2_overall = r2_score(all_targets[mask_all], all_preds[mask_all])
                except:
                    r2_overall = 0.0
            else:
                r2_overall = 0.0

        loss_components = {
            'total': total_loss.item(),
            'supervised': supervised_loss.item()
        }

        metrics = {
            'mse': mse,
            'mae': mae,
            'r2_p': r2_p,
            'r2_u': r2_u,
            'r2_v': r2_v,
            'r2_overall': r2_overall
        }

        return total_loss, loss_components, metrics

    def predict(self, inlet_speed: float, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:

        self.eval()

        if isinstance(x, (int, float)):
            x = np.array([x])
        if isinstance(y, (int, float)):
            y = np.array([y])
        if isinstance(t, (int, float)):
            t = np.array([t])

        n_points = len(x)
        inlet_speeds = np.full((n_points, 1), inlet_speed, dtype=np.float32)
        coordinates = np.column_stack([x, y, t]).astype(np.float32)

        inlet_speeds_tensor = torch.FloatTensor(inlet_speeds).to(self.device)
        coordinates_tensor = torch.FloatTensor(coordinates).to(self.device)

        with torch.no_grad():
            outputs = self.forward(inlet_speeds_tensor, coordinates_tensor)
            outputs_np = outputs.cpu().numpy()

        p_pred = outputs_np[:, 0]
        u_pred = outputs_np[:, 1]
        v_pred = outputs_np[:, 2]

        return {
            'p': p_pred,
            'u': u_pred,
            'v': v_pred,
            'x': x,
            'y': y,
            't': t,
            'inlet_speed': inlet_speed
        }

    def predict_batch(self, inlet_speeds: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        self.eval()


        inlet_speeds_tensor = torch.FloatTensor(inlet_speeds).to(self.device)
        coordinates_tensor = torch.FloatTensor(coordinates).to(self.device)

        with torch.no_grad():
            outputs = self.forward(inlet_speeds_tensor, coordinates_tensor)
            outputs_np = outputs.cpu().numpy()

        return outputs_np

    def evaluate(self, dataloader, device=None) -> Dict[str, float]:

        if device is None:
            device = self.device

        self.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch_data in dataloader:
                inlet_speeds, inputs, outputs, data_types, _ = batch_data

                inlet_speeds = inlet_speeds.to(device)
                inputs = inputs.to(device)
                outputs = outputs.to(device)
                data_types = data_types.to(device)

                supervised_mask = (data_types == 0).squeeze()
                if supervised_mask.any():
                    sup_inlet = inlet_speeds[supervised_mask]
                    sup_inputs = inputs[supervised_mask]
                    sup_targets = outputs[supervised_mask]

                    sup_preds = self.forward(sup_inlet, sup_inputs)

                    loss = F.mse_loss(sup_preds, sup_targets)
                    total_loss += loss.item()

                    all_predictions.append(sup_preds.cpu().numpy())
                    all_targets.append(sup_targets.cpu().numpy())

                batch_count += 1

        if len(all_predictions) > 0:
            all_predictions = np.vstack(all_predictions)
            all_targets = np.vstack(all_targets)

            metrics = {
                'mse': float(mean_squared_error(all_targets, all_predictions)),
                'mae': float(np.mean(np.abs(all_targets - all_predictions))),
                'r2_p': float(r2_score(all_targets[:, 0], all_predictions[:, 0])),
                'r2_u': float(r2_score(all_targets[:, 1], all_predictions[:, 1])),
                'r2_v': float(r2_score(all_targets[:, 2], all_predictions[:, 2])),
                'r2_overall': float(r2_score(all_targets.reshape(-1), all_predictions.reshape(-1))),
                'avg_loss': total_loss / batch_count
            }
        else:
            metrics = {
                'mse': 0.0,
                'mae': 0.0,
                'r2_p': 0.0,
                'r2_u': 0.0,
                'r2_v': 0.0,
                'r2_overall': 0.0,
                'avg_loss': 0.0
            }

        return metrics

    def save_model(self, filepath: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'training_stage': self.training_stage,
        }, filepath)
        print(f"model save: {filepath}")

    def load_model(self, filepath: str, device=None):
        if device is None:
            device = self.device

        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_stage = checkpoint.get('training_stage', 'stage1')
        print(f"model from {filepath} load")
        print(f"training stage: {self.training_stage}")


class DataDrivenTrainer:

    def __init__(self, model: DataDrivenNeuralOperator, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'train_mae': [],
            'val_mae': [],
            'train_r2_p': [],
            'val_r2_p': [],
            'train_r2_u': [],
            'val_r2_u': [],
            'train_r2_v': [],
            'val_r2_v': [],
            'train_r2_overall': [],
            'val_r2_overall': [],
            'learning_rates': []
        }

        self.log_rows = []

    def train_epoch(self, train_loader, epoch: int, total_epochs: int) -> Dict[str, float]:

        self.model.train()
        total_loss = 0.0
        batch_count = 0

        epoch_metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'r2_p': 0.0,
            'r2_u': 0.0,
            'r2_v': 0.0,
            'r2_overall': 0.0
        }
        supervised_batch_count = 0

        for batch_idx, batch_data in enumerate(train_loader):

            self.optimizer.zero_grad()

            loss, loss_components, metrics = self.model.compute_loss(batch_data)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if loss_components.get('supervised', 0) > 0:
                supervised_batch_count += 1
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key] += metrics[key]

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}, "
                      f"R²: {metrics.get('r2_overall', 0):.4f}")

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        for key in epoch_metrics:
            epoch_metrics[key] = epoch_metrics[key] / supervised_batch_count if supervised_batch_count > 0 else 0.0

        return {
            'loss': avg_loss,
            'metrics': epoch_metrics
        }

    def validate(self, val_loader) -> Dict[str, float]:

        return self.model.evaluate(val_loader)

    def save_training_log_to_excel(self, save_path: str):

        if not self.log_rows:
            print("⚠️ none")
            return None

        df = pd.DataFrame(self.log_rows)

        if not save_path.endswith('.xlsx'):
            save_path = f"{save_path}_training_log.xlsx"

        try:
            df.to_excel(save_path, index=False)
            print(f"✅ training_log_to_excel: {save_path}")

            csv_path = save_path.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"✅ training_log_to_CSV: {csv_path}")

            return df, save_path, csv_path
        except Exception as e:
            print(f"❌ save failed: {e}")
            return None

    def train(self, train_loader, val_loader, num_epochs: int,
              save_path: str = None, early_stopping_patience: int = None,
              save_log: bool = True, log_interval: int = 1):

        print(f"\n begin training {num_epochs} epoch")
        print("=" * 60)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            train_stats = self.train_epoch(train_loader, epoch, num_epochs)

            val_stats = self.validate(val_loader)

            self.history['train_loss'].append(train_stats['loss'])
            self.history['val_loss'].append(val_stats['avg_loss'])
            self.history['train_mse'].append(train_stats['metrics']['mse'])
            self.history['val_mse'].append(val_stats['mse'])
            self.history['train_mae'].append(train_stats['metrics']['mae'])
            self.history['val_mae'].append(val_stats['mae'])
            self.history['train_r2_p'].append(train_stats['metrics']['r2_p'])
            self.history['val_r2_p'].append(val_stats['r2_p'])
            self.history['train_r2_u'].append(train_stats['metrics']['r2_u'])
            self.history['val_r2_u'].append(val_stats['r2_u'])
            self.history['train_r2_v'].append(train_stats['metrics']['r2_v'])
            self.history['val_r2_v'].append(val_stats['r2_v'])
            self.history['train_r2_overall'].append(train_stats['metrics']['r2_overall'])
            self.history['val_r2_overall'].append(val_stats['r2_overall'])

            if self.scheduler is not None:
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            if epoch % log_interval == 0 or epoch == num_epochs - 1:
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0

                log_entry = {
                    'epoch': epoch + 1,
                    'train_loss': train_stats['loss'],
                    'val_loss': val_stats['avg_loss'],
                    'train_mse': train_stats['metrics']['mse'],
                    'val_mse': val_stats['mse'],
                    'train_mae': train_stats['metrics']['mae'],
                    'val_mae': val_stats['mae'],
                    'train_r2_p': train_stats['metrics']['r2_p'],
                    'val_r2_p': val_stats['r2_p'],
                    'train_r2_u': train_stats['metrics']['r2_u'],
                    'val_r2_u': val_stats['r2_u'],
                    'train_r2_v': train_stats['metrics']['r2_v'],
                    'val_r2_v': val_stats['r2_v'],
                    'train_r2_overall': train_stats['metrics']['r2_overall'],
                    'val_r2_overall': val_stats['r2_overall'],
                    'learning_rate': current_lr
                }

                self.log_rows.append(log_entry)

            print(f"train loss: {train_stats['loss']:.6f}")
            print(f"valid loss: {val_stats['avg_loss']:.6f}")
            print(f"R²: p={val_stats['r2_p']:.4f}, u={val_stats['r2_u']:.4f}, v={val_stats['r2_v']:.4f}, "
                  f"overall={val_stats['r2_overall']:.4f}")
            print(f"MSE: {val_stats['mse']:.6f}")


            if val_stats['avg_loss'] < best_val_loss and save_path is not None:
                best_val_loss = val_stats['avg_loss']
                self.model.save_model(f"{save_path}_best.pth")
                print(f"best_val_loss (val_loss={best_val_loss:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1

            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"early_stopping {early_stopping_patience}")
                break

        if save_path is not None:
            self.model.save_model(f"{save_path}_final.pth")

            if save_log and self.log_rows:
                self.save_training_log_to_excel(save_path)

        print(f"\ntraining complete!")
        print(f"best_val: {best_val_loss:.6f}")
        print("=" * 60)

    def plot_training_history(self, save_path: str = None):

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))


        if not self.history['train_loss']:
            print("⚠️  no history")
            return

        epochs = range(1, len(self.history['train_loss']) + 1)

        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validating Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')


        axes[0, 1].plot(epochs, self.history['train_mse'], 'b-', label='Training')
        axes[0, 1].plot(epochs, self.history['val_mse'], 'r-', label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].set_title('Training and Validating MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')

        axes[0, 2].plot(epochs, self.history['train_mae'], 'b-', label='Training')
        axes[0, 2].plot(epochs, self.history['val_mae'], 'r-', label='Validation')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].set_title('Training and Validating MAE')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)


        axes[1, 0].plot(epochs, self.history['train_r2_p'], 'r-', label='Train R²(P)')
        axes[1, 0].plot(epochs, self.history['train_r2_u'], 'g-', label='Train R²(U)')
        axes[1, 0].plot(epochs, self.history['train_r2_v'], 'b-', label='Train R²(V)')
        axes[1, 0].plot(epochs, self.history['train_r2_overall'], 'm-', label='Train R²(Overall)', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('Training R² Score Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([-0.1, 1.1])

        axes[1, 1].plot(epochs, self.history['val_r2_p'], 'r-', label='Val R²(P)')
        axes[1, 1].plot(epochs, self.history['val_r2_u'], 'g-', label='Val R²(U)')
        axes[1, 1].plot(epochs, self.history['val_r2_v'], 'b-', label='Val R²(V)')
        axes[1, 1].plot(epochs, self.history['val_r2_overall'], 'm-', label='Val R²(Overall)', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].set_title('Validating R² Score Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([-0.1, 1.1])

        axes[1, 2].plot(epochs, self.history['train_r2_overall'], 'b-', label='Training')
        axes[1, 2].plot(epochs, self.history['val_r2_overall'], 'r-', label='Validation')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('R²')
        axes[1, 2].set_title('Training vs Validating R²')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim([-0.1, 1.1])

        axes[2, 0].plot(epochs, self.history['train_r2_p'], 'b-', label='Train R²(P)')
        axes[2, 0].plot(epochs, self.history['val_r2_p'], 'r--', label='Val R²(P)')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('R²')
        axes[2, 0].set_title('Pressure (p) R² Score')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_ylim([-0.1, 1.1])

        axes[2, 1].plot(epochs, self.history['train_r2_u'], 'b-', label='Train R²(U)')
        axes[2, 1].plot(epochs, self.history['val_r2_u'], 'r--', label='Val R²(U)')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('R²')
        axes[2, 1].set_title('Velocity U R² Score')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylim([-0.1, 1.1])

        axes[2, 2].plot(epochs, self.history['train_r2_v'], 'b-', label='Train R²(V)')
        axes[2, 2].plot(epochs, self.history['val_r2_v'], 'r--', label='Val R²(V)')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('R²')
        axes[2, 2].set_title('Velocity V R² Score')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].set_ylim([-0.1, 1.1])

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"history fig: {save_path}")

        plt.show()
        plt.close()

    def export_history_to_excel(self, save_path: str = "training_history.xlsx"):

        history_df = pd.DataFrame({
            'epoch': list(range(1, len(self.history['train_loss']) + 1)),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'train_mse': self.history['train_mse'],
            'val_mse': self.history['val_mse'],
            'train_mae': self.history['train_mae'],
            'val_mae': self.history['val_mae'],
            'train_r2_p': self.history['train_r2_p'],
            'val_r2_p': self.history['val_r2_p'],
            'train_r2_u': self.history['train_r2_u'],
            'val_r2_u': self.history['val_r2_u'],
            'train_r2_v': self.history['train_r2_v'],
            'val_r2_v': self.history['val_r2_v'],
            'train_r2_overall': self.history['train_r2_overall'],
            'val_r2_overall': self.history['val_r2_overall']
        })

        if self.history['learning_rates']:
            history_df['learning_rate'] = self.history['learning_rates']

        try:
            history_df.to_excel(save_path, index=False)
            print(f"✅ history_df.to_excel: {save_path}")

            csv_path = save_path.replace('.xlsx', '.csv')
            history_df.to_csv(csv_path, index=False)
            print(f"✅ history_df.to_CSV: {csv_path}")

            return history_df, save_path, csv_path
        except Exception as e:
            print(f"❌ export error: {e}")
            return None



def test_data_driven_model():
    print("=" * 60)
    print("Ref ")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    model = DataDrivenNeuralOperator(
        branch_layers=(1, 64, 64, 64),
        trunk_layers=(3, 64, 64, 64),
        output_dim=3,
        device=device
    )

    print(f"\nmodel.parameters: {sum(p.numel() for p in model.parameters()):,}")


    batch_size = 4
    inlet_speeds = torch.randn(batch_size, 1).to(device)
    coordinates = torch.randn(batch_size, 3).to(device)

    outputs = model.forward(inlet_speeds, coordinates)
    print(f"inlet shape: 速度={inlet_speeds.shape}, 坐标={coordinates.shape}")
    print(f"outputs shape: {outputs.shape}")
    print(f"output: p={outputs[0, 0]:.6f}, u={outputs[0, 1]:.6f}, v={outputs[0, 2]:.6f}")


    data_types = torch.zeros(batch_size, 1).to(device)
    targets = torch.randn(batch_size, 3).to(device)

    batch_data = (inlet_speeds, coordinates, targets, data_types, torch.zeros(batch_size, 1).to(device))
    loss, loss_components, metrics = model.compute_loss(batch_data)

    print(f"loss: {loss.item():.6f}")
    print(f"loss_components: {loss_components}")
    print(f"Evaluation Metrics: {metrics}")


    inlet_speed = 0.1
    x = np.array([0.1, 0.2, 0.3])
    y = np.array([0.0, 0.1, 0.0])
    t = np.array([1.0, 2.0, 3.0])

    predictions = model.predict(inlet_speed, x, y, t)
    for i in range(len(x)):
        print(f"  {i}: x={x[i]:.2f}, y={y[i]:.2f}, t={t[i]:.1f}, "
              f"p={predictions['p'][i]:.6f}, u={predictions['u'][i]:.6f}, v={predictions['v'][i]:.6f}")

    print("\n model completed!")


if __name__ == "__main__":
    test_data_driven_model()