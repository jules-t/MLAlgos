import torch
from torch.utils.data import Dataset
import numpy as np


class SwissRollDataset(Dataset):
    """2D Swiss Roll dataset for diffusion model training."""

    def __init__(self, n_samples: int = 10000, noise: float = 0.1, seed: int = 42):
        """
        Generate a 2D Swiss Roll dataset.

        Args:
            n_samples: Number of points to generate
            noise: Standard deviation of Gaussian noise added to the data
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        # Generate random angles between 1π and 4π
        phi = np.random.uniform(1 * np.pi, 4 * np.pi, n_samples)

        # Radius increases with angle
        r = phi

        # Convert polar to Cartesian coordinates
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        # Stack into (n_samples, 2) array
        data = np.stack([x, y], axis=1)

        # Add noise
        data += np.random.randn(n_samples, 2) * noise

        # Normalize to roughly [-1, 1] range
        data = (data - data.mean(axis=0)) / data.std(axis=0)

        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
