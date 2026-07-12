import numpy as np
import os
import random
from typing import List, Optional, Dict, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class PhysicsOnlyDataset(Dataset):

    def __init__(self,
                 domain_bounds: Dict = None,
                 speeds_range: Tuple[float, float] = (0.073, 0.365),  # 速度范围
                 time_range: Tuple[int, int] = (0, 20),  # 时间范围
                 num_points: int = 10000,  # 总点数
                 device: str = 'cpu',
                 normalizer: Optional[object] = None,
                 cylinder_radius: float = 0.005,
                 seed: int = 42):

        self.device = device
        self.normalizer = normalizer
        self.cylinder_radius = cylinder_radius
        self.num_points = num_points


        np.random.seed(seed)
        random.seed(seed)


        self.domain_bounds = domain_bounds or {
            'x_min': -0.1, 'x_max': 0.3,
            'y_min': -0.1, 'y_max': 0.1
        }


        self.speed_min, self.speed_max = speeds_range


        self.time_min, self.time_max = time_range


        self.inputs = []  #  [x, y, t, U_in]


        self._generate_pde_points()


        self._prepare_data()


        self._print_statistics()

    def _generate_pde_points(self):

        print(f"\n generate {self.num_points} 个PDE...")

        points_generated = 0
        max_attempts = self.num_points * 10

        pbar = tqdm(total=self.num_points, desc="generatePDE")

        while points_generated < self.num_points and len(self.inputs) < max_attempts:

            x = np.random.uniform(self.domain_bounds['x_min'], self.domain_bounds['x_max'])
            y = np.random.uniform(self.domain_bounds['y_min'], self.domain_bounds['y_max'])


            distance_from_origin = np.sqrt(x ** 2 + y ** 2)
            if distance_from_origin <= self.cylinder_radius:
                continue


            t = np.random.uniform(self.time_min, self.time_max)


            speed = np.random.uniform(self.speed_min, self.speed_max)


            features = [x, y, t, speed]

            self.inputs.append(features)
            points_generated += 1
            pbar.update(1)

        pbar.close()

        if points_generated < self.num_points:
            print(f"Warning: Only {points_generated} ，less than {self.num_points}")

    def _prepare_data(self):

        if len(self.inputs) == 0:
            self.inputs_tensor = torch.tensor([], dtype=torch.float32)
            return


        inputs_array = np.array(self.inputs, dtype=np.float32)


        if self.normalizer and hasattr(self.normalizer, '_initialized') and self.normalizer._initialized:
            if hasattr(self.normalizer, 'transform_inputs'):
                inputs_array = self.normalizer.transform_inputs(inputs_array)


        self.inputs_tensor = torch.from_numpy(inputs_array).to(self.device)

    def _print_statistics(self):

        print("\n" + "=" * 60)
        print("Statistical analysis of physical equation data sets")
        print("=" * 60)

        print(f"total: {len(self.inputs_tensor):,}")
        print(f"Spatial scope: x∈[{self.domain_bounds['x_min']}, {self.domain_bounds['x_max']}], "
              f"y∈[{self.domain_bounds['y_min']}, {self.domain_bounds['y_max']}]")
        print(f"time range: t∈[{self.time_min}, {self.time_max}]")
        print(f"velocity range: U_in∈[{self.speed_min:.3f}, {self.speed_max:.3f}]")
        print(f"cylinder radius: {self.cylinder_radius}")

        if len(self.inputs_tensor) > 0:
            inputs_np = self.inputs_tensor.cpu().numpy()
            print(f"\nInput statistics")
            print(f"  x: [{inputs_np[:, 0].min():.3f}, {inputs_np[:, 0].max():.3f}]")
            print(f"  y: [{inputs_np[:, 1].min():.3f}, {inputs_np[:, 1].max():.3f}]")
            print(f"  t: [{inputs_np[:, 2].min():.1f}, {inputs_np[:, 2].max():.1f}]")
            print(f"  U_in: [{inputs_np[:, 3].min():.3f}, {inputs_np[:, 3].max():.3f}]")

    def __len__(self) -> int:
        return len(self.inputs_tensor)

    def __getitem__(self, idx: int):
        return self.inputs_tensor[idx]


def create_physics_only_dataloader(
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        domain_bounds: Dict = None,
        speeds_range: Tuple[float, float] = (0.073, 0.365),
        time_range: Tuple[int, int] = (0, 20),
        num_points: int = 10000,
        device: str = 'cpu',
        normalizer: Optional[object] = None,
        cylinder_radius: float = 0.005,
        seed: int = 42
) -> DataLoader:


    dataset = PhysicsOnlyDataset(
        domain_bounds=domain_bounds,
        speeds_range=speeds_range,
        time_range=time_range,
        num_points=num_points,
        device=device,
        normalizer=normalizer,
        cylinder_radius=cylinder_radius,
        seed=seed
    )

    def collate_fn(batch):

        inputs_batch = torch.stack(batch)
        return inputs_batch

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )

    return dataloader


# testt function
def test_physics_only_dataloader():
    print("=" * 60)
    print("Test the data loader for physical equations")
    print("=" * 60)

    normalizer_path = r"experiments\stage1_20260413_105850\models\normalizer.pth"
    normalizer = load_normalizer_from_stage1(normalizer_path)

    if normalizer is None:
        print("Warning: The standardizer cannot be loaded. Using unstandardized data instead")

    print("\n Create a data loader for physical equations...")
    try:
        dataloader = create_physics_only_dataloader(
            batch_size=16,
            shuffle=True,
            domain_bounds={
                'x_min': -0.1, 'x_max': 0.3,
                'y_min': -0.1, 'y_max': 0.1
            },
            speeds_range=(0.073, 0.365),
            time_range=(0, 20),
            num_points=1000,
            device='cpu',
            normalizer=normalizer,
            cylinder_radius=0.005,
            seed=42
        )

        print(f"dataset size: {len(dataloader.dataset)}")
        print(f"batch: {len(dataloader)}")

        for batch_idx, inputs_batch in enumerate(dataloader):
            print(f"\nbatch {batch_idx}:")
            print(f"  inout shape: {inputs_batch.shape}")  # [batch_size, 4]

            print(f"\n  input range:")
            print(f"    x: [{inputs_batch[:, 0].min():.3f}, {inputs_batch[:, 0].max():.3f}]")
            print(f"    y: [{inputs_batch[:, 1].min():.3f}, {inputs_batch[:, 1].max():.3f}]")
            print(f"    t: [{inputs_batch[:, 2].min():.1f}, {inputs_batch[:, 2].max():.1f}]")
            print(f"    U_in: [{inputs_batch[:, 3].min():.3f}, {inputs_batch[:, 3].max():.3f}]")

            print(f"\n  first:")
            print(f"    input: x={inputs_batch[0, 0]:.3f}, y={inputs_batch[0, 1]:.3f}, "
                  f"t={inputs_batch[0, 2]:.1f}, U_in={inputs_batch[0, 3]:.3f}")

            break

        print("\ntest completed!")

    except Exception as e:
        print(f"failed: {e}")
        import traceback
        traceback.print_exc()


def load_normalizer_from_stage1(normalizer_path: str) -> Optional[object]:
    try:
        checkpoint = torch.load(normalizer_path, map_location='cpu')

        from stage1_dataloader import DataNormalizer
        normalizer = DataNormalizer('zscore')

        normalizer.method = checkpoint.get('method', 'zscore')
        normalizer.input_stats = checkpoint.get('input_stats', {})
        normalizer.output_stats = checkpoint.get('output_stats', {})
        normalizer._initialized = checkpoint.get('_initialized', False)

        return normalizer
    except Exception as e:
        print(f"Loading the standardizer failed.: {e}")
        return None



if __name__ == "__main__":

    test_physics_only_dataloader()