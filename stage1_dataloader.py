import numpy as np
import os
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import warnings


class DataNormalizer:

    def __init__(self, method='zscore'):
        self.method = method
        self.input_stats = {}
        self.output_stats = {}
        self._initialized = False

    def fit(self, inputs, outputs):

        if self.method == 'none':
            self._initialized = True
            return

        if inputs.size > 0:
            for i in range(inputs.shape[1]):
                if i == 0:  # x
                    col_name = 'x'
                elif i == 1:  # y
                    col_name = 'y'
                elif i == 2:  # t
                    col_name = 't'
                else:
                    col_name = 'U_in'

                self.input_stats[col_name] = {
                    'mean': np.mean(inputs[:, i]),
                    'std': np.std(inputs[:, i]) + 1e-8
                }

        if outputs.size > 0:
            output_names = ['p', 'u', 'v']
            for i in range(outputs.shape[1]):
                col_name = output_names[i]
                self.output_stats[col_name] = {
                    'mean': np.mean(outputs[:, i]),
                    'std': np.std(outputs[:, i]) + 1e-8
                }

        self._initialized = True
        for col, stats in self.input_stats.items():
            print(f"  {col}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
        for col, stats in self.output_stats.items():
            print(f"  {col}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

    def transform_inputs(self, inputs):
        if self.method == 'none' or not self._initialized or len(self.input_stats) == 0:
            return inputs

        normalized = inputs.copy()
        for i in range(inputs.shape[1]):
            if i == 0:  # x
                col_name = 'x'
            elif i == 1:  # y
                col_name = 'y'
            elif i == 2:  # t
                col_name = 't'
            else:
                col_name = 'U_in'

            if col_name in self.input_stats:
                mean = self.input_stats[col_name]['mean']
                std = self.input_stats[col_name]['std']
                normalized[:, i] = (inputs[:, i] - mean) / std

        return normalized

    def transform_outputs(self, outputs):
        if self.method == 'none' or not self._initialized or len(self.output_stats) == 0:
            return outputs

        normalized = outputs.copy()
        output_names = ['p', 'u', 'v']
        for i in range(outputs.shape[1]):
            col_name = output_names[i]
            if col_name in self.output_stats:
                mean = self.output_stats[col_name]['mean']
                std = self.output_stats[col_name]['std']
                normalized[:, i] = (outputs[:, i] - mean) / std

        return normalized

    def inverse_transform_outputs(self, outputs_normalized):
        if self.method == 'none' or not self._initialized or len(self.output_stats) == 0:
            return outputs_normalized

        original = outputs_normalized.copy()
        output_names = ['p', 'u', 'v']
        for i in range(outputs_normalized.shape[1]):
            col_name = output_names[i]
            if col_name in self.output_stats:
                mean = self.output_stats[col_name]['mean']
                std = self.output_stats[col_name]['std']
                original[:, i] = outputs_normalized[:, i] * std + mean

        return original

    def save_stats(self, filepath):
        stats = {
            'method': self.method,
            'input_stats': self.input_stats,
            'output_stats': self.output_stats,
            '_initialized': self._initialized
        }
        torch.save(stats, filepath)
        print(f"save: {filepath}")

    def load_stats(self, filepath):
        stats = torch.load(filepath)
        self.method = stats.get('method', 'zscore')
        self.input_stats = stats.get('input_stats', {})
        self.output_stats = stats.get('output_stats', {})
        self._initialized = stats.get('_initialized', False)
        print(f"from {filepath} load")
        return self


class DataDrivenDataset(Dataset):

    def __init__(self,
                 data_dir: str = 'supervision_data',
                 speeds: Optional[List[float]] = None,
                 time_steps: Optional[List[int]] = None,
                 device: str = 'cpu',
                 normalize: bool = True,
                 add_inlet_speed: bool = True,
                 shuffle: bool = True):

        self.data_dir = data_dir
        self.device = device
        self.normalize = normalize
        self.add_inlet_speed = add_inlet_speed
        self.shuffle = shuffle

        self.all_speeds = self._get_available_speeds()
        self.speeds = speeds if speeds is not None else self.all_speeds

        self.time_steps = time_steps if time_steps is not None else list(range(1, 21))

        self.inputs = []
        self.outputs = []
        self.raw_inputs = []
        self.raw_outputs = []

        self.normalizer = DataNormalizer('zscore')

        self._load_data()

        if len(self.inputs) > 0:
            self.inputs = np.array(self.inputs, dtype=np.float32)
            self.outputs = np.array(self.outputs, dtype=np.float32)
            self.raw_inputs = np.array(self.raw_inputs, dtype=np.float32)
            self.raw_outputs = np.array(self.raw_outputs, dtype=np.float32)

            if self.shuffle and len(self.inputs) > 1:
                indices = np.random.permutation(len(self.inputs))
                self.inputs = self.inputs[indices]
                self.outputs = self.outputs[indices]
                self.raw_inputs = self.raw_inputs[indices]
                self.raw_outputs = self.raw_outputs[indices]

            if self.normalize:
                self.normalizer.fit(self.raw_inputs, self.raw_outputs)
                self.inputs = self.normalizer.transform_inputs(self.inputs)
                self.outputs = self.normalizer.transform_outputs(self.outputs)

            print(f"\n数据集统计信息:")
            print(f"  total sample: {len(self.inputs):,}")
            print(f"  input dimension: {self.inputs.shape[1]}")
            print(f"  output dimension: {self.outputs.shape[1]}")
            print(f"  velocity: {len(np.unique(self.raw_inputs[:, 3] if self.add_inlet_speed else []))} 个不同值")
            print(f"  time range: {np.min(self.raw_inputs[:, 2]):.0f} - {np.max(self.raw_inputs[:, 2]):.0f}")
            if self.normalize:
                print(f" open")
        else:
            warnings.warn("entry！")
            self.inputs = np.array([], dtype=np.float32)
            self.outputs = np.array([], dtype=np.float32)
            self.raw_inputs = np.array([], dtype=np.float32)
            self.raw_outputs = np.array([], dtype=np.float32)

    def _get_available_speeds(self) -> List[float]:
        speeds = []
        if not os.path.exists(self.data_dir):
            return speeds

        for filename in os.listdir(self.data_dir):
            if filename.startswith('data_U_') and (filename.endswith('.npy') or filename.endswith('.txt')):
                try:
                    speed_str = filename.split('_')[2].split('.')[0]
                    speed = float(speed_str.replace('_', '.'))
                    speeds.append(speed)
                except (IndexError, ValueError):
                    continue

        return sorted(speeds)

    def _load_data(self):

        total_samples = 0
        for speed in self.speeds:
            speed_str = f"{speed:.3f}".replace('.', '_')
            npy_file = os.path.join(self.data_dir, f'data_U_{speed_str}.npy')
            txt_file = os.path.join(self.data_dir, f'data_U_{speed_str}.txt')

            data = None
            if os.path.exists(npy_file):
                data = np.load(npy_file)
                print(f"  load: {npy_file} ({len(data)} 个样本)")
            elif os.path.exists(txt_file):
                data = np.loadtxt(txt_file, delimiter=',', skiprows=1)
                print(f"  load: {txt_file} ({len(data)} 个样本)")
            else:
                print(f"  can not find {speed:.3f} data")
                continue

            if data is None or len(data) == 0:
                continue

            for i in range(len(data)):
                x, y, t, p, u, v = data[i]

                if int(t) in self.time_steps:

                    if self.add_inlet_speed:
                        inputs = [x, y, t, speed]
                    else:
                        inputs = [x, y, t]

                    outputs = [p, u, v]

                    self.inputs.append(inputs)
                    self.outputs.append(outputs)
                    self.raw_inputs.append(inputs)
                    self.raw_outputs.append(outputs)
                    total_samples += 1

        print(f"total sample: {total_samples}")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.FloatTensor(self.inputs[idx]).to(self.device)
        outputs = torch.FloatTensor(self.outputs[idx]).to(self.device)
        return inputs, outputs

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.raw_inputs[idx], self.raw_outputs[idx]

    def get_normalized_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.inputs[idx], self.outputs[idx]

    def get_statistics(self) -> dict:
        if len(self.inputs) == 0:
            return {}

        stats = {
            'total_samples': len(self.inputs),
            'input_dim': self.inputs.shape[1],
            'output_dim': self.outputs.shape[1],
            'input_stats': self.normalizer.input_stats if self.normalizer._initialized else {},
            'output_stats': self.normalizer.output_stats if self.normalizer._initialized else {},
        }

        if len(self.raw_inputs) > 0:
            stats['speed_range'] = {
                'min': float(np.min(self.raw_inputs[:, 3])) if self.add_inlet_speed else None,
                'max': float(np.max(self.raw_inputs[:, 3])) if self.add_inlet_speed else None,
                'unique': len(np.unique(self.raw_inputs[:, 3])) if self.add_inlet_speed else 0
            }

            stats['time_range'] = {
                'min': float(np.min(self.raw_inputs[:, 2])),
                'max': float(np.max(self.raw_inputs[:, 2])),
                'unique': len(np.unique(self.raw_inputs[:, 2]))
            }

            stats['spatial_range'] = {
                'x_min': float(np.min(self.raw_inputs[:, 0])),
                'x_max': float(np.max(self.raw_inputs[:, 0])),
                'y_min': float(np.min(self.raw_inputs[:, 1])),
                'y_max': float(np.max(self.raw_inputs[:, 1]))
            }

        return stats


def create_dataloader(
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        data_dir: str = 'supervision_data',
        speeds: Optional[List[float]] = None,
        time_steps: Optional[List[int]] = None,
        device: str = 'cpu',
        normalize: bool = True,
        add_inlet_speed: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, DataNormalizer]:

    np.random.seed(seed)

    full_dataset = DataDrivenDataset(
        data_dir=data_dir,
        speeds=speeds,
        time_steps=time_steps,
        device=device,
        normalize=normalize,
        add_inlet_speed=add_inlet_speed,
        shuffle=shuffle
    )

    if len(full_dataset) == 0:
        raise ValueError("error")

    normalizer = full_dataset.normalizer

    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    print(f"\ndata split:")
    print(f"  sample: {n_total}")
    print(f"  train: {n_train} ({train_ratio * 100:.1f}%)")
    print(f"  valid: {n_val} ({val_ratio * 100:.1f}%)")
    print(f"  test: {n_test} ({test_ratio * 100:.1f}%)")

    if full_dataset.shuffle:
        train_dataset = torch.utils.data.Subset(full_dataset, range(0, n_train))
        val_dataset = torch.utils.data.Subset(full_dataset, range(n_train, n_train + n_val))
        test_dataset = torch.utils.data.Subset(full_dataset, range(n_train + n_val, n_total))
    else:
        indices = np.arange(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    def collate_fn(batch):
        inputs, outputs = zip(*batch)
        inputs = torch.stack(inputs)
        outputs = torch.stack(outputs)
        return inputs, outputs

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )

    return train_loader, val_loader, test_loader, normalizer


def create_simple_dataloader(
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        data_dir: str = 'supervision_data',
        speeds: Optional[List[float]] = None,
        time_steps: Optional[List[int]] = None,
        device: str = 'cpu',
        normalize: bool = True,
        add_inlet_speed: bool = True
) -> DataLoader:

    dataset = DataDrivenDataset(
        data_dir=data_dir,
        speeds=speeds,
        time_steps=time_steps,
        device=device,
        normalize=normalize,
        add_inlet_speed=add_inlet_speed,
        shuffle=shuffle
    )

    def collate_fn(batch):
        inputs, outputs = zip(*batch)
        inputs = torch.stack(inputs)
        outputs = torch.stack(outputs)
        return inputs, outputs

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )



def test_data_loading():
    print("=" * 60)
    print("=" * 60)


    if not os.path.exists('supervision_data'):
        return


    files = os.listdir('supervision_data')
    print(f"supervised data ({len(files)} 个):")
    for file in sorted(files)[:10]:
        print(f"  {file}")
    if len(files) > 10:
        print(f"  ... left {len(files) - 10} ")

    test_speeds = [0.095, 0.146, 0.219, 0.292]
    print(f"\ntest : {[f'{s:.3f}' for s in test_speeds]}")

    print("\n" + "=" * 60)
    print("=" * 60)

    try:
        dataloader = create_simple_dataloader(
            batch_size=16,
            shuffle=True,
            data_dir='supervision_data',
            speeds=test_speeds,
            time_steps=list(range(1, 6)),
            device='cpu',
            normalize=True,
            add_inlet_speed=True
        )

        print(f"dataset size: {len(dataloader.dataset)}")

        if len(dataloader.dataset) > 0:
            stats = dataloader.dataset.get_statistics()
            if stats:
                print(f"  total: {stats['total_samples']}")
                print(f"  input_dim: {stats['input_dim']}")
                print(f"  output_dim: {stats['output_dim']}")
                if 'speed_range' in stats and stats['speed_range']['unique'] > 0:
                    print(f"  speed_range: {stats['speed_range']['min']:.3f} - {stats['speed_range']['max']:.3f}")
                if 'time_range' in stats:
                    print(f"  time_range: {stats['time_range']['min']:.0f} - {stats['time_range']['max']:.0f}")

            for batch_idx, (inputs_batch, outputs_batch) in enumerate(dataloader):
                print(f"\nbatch {batch_idx}:")
                print(f"  inputs_batch.shape: {inputs_batch.shape}")
                print(f"  outputs_batch.shape: {outputs_batch.shape}")

                print(f"\nfirst:")
                input_names = ['x', 'y', 't', 'U_in'] if inputs_batch.shape[1] == 4 else ['x', 'y', 't']
                for i, name in enumerate(input_names):
                    print(f"  {name}: {inputs_batch[0, i].item():.6f}")

                output_names = ['p', 'u', 'v']
                for i, name in enumerate(output_names):
                    print(f"  {name}: {outputs_batch[0, i].item():.6f}")

                break
        else:
            print("entry!")
    except Exception as e:
        print(f" load error {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("=" * 60)

    try:
        train_loader, val_loader, test_loader, normalizer = create_dataloader(
            batch_size=16,
            shuffle=True,
            data_dir='supervision_data',
            speeds=test_speeds,
            time_steps=list(range(1, 6)),
            device='cpu',
            normalize=True,
            add_inlet_speed=True,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

        print(f"train: {len(train_loader.dataset)} ")
        print(f"valid: {len(val_loader.dataset)} ")
        print(f"test: {len(test_loader.dataset)} ")

        for batch_idx, (inputs_batch, outputs_batch) in enumerate(train_loader):
            print(f"\nbatch {batch_idx}:")
            print(f"  inputs_batch.shape: {inputs_batch.shape}")
            print(f"  outputs_batch.shape: {outputs_batch.shape}")
            break

    except Exception as e:
        import traceback
        traceback.print_exc()

    print("\ntest complete!")



if __name__ == "__main__":
    test_data_loading()