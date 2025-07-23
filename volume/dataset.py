import torch
import numpy as np
from torch.utils.data import Dataset

"""
Dataset Class for training network
mapping latent vectors to volumes
"""
class VolumeDataset(Dataset):

    def __init__(self, dataset_path):
        """
            Expecting a npz file with two arrays:
            latents and volumes
        """
        self.file_path = dataset_path
        data = np.load(dataset_path)
        self.latents = torch.tensor(data["latents"], dtype=torch.float32)
        self.volumes = torch.tensor(data["volumes"], dtype=torch.float32)
        self.size = self.latents.shape[0]

    def __getitem__(self, idx):
        return self.latents[idx], self.volumes[idx]

    def collate_fn(self, batch):
        latents, volumes = zip(*batch)
        return torch.stack(latents), torch.stack(volumes)

    def __len__(self):
        return self.size
    
class GeneraDataset(Dataset):

    def __init__(self, dataset_path):
        """
            Expecting a npz file with two arrays:
            latents and genera
        """
        self.file_path = dataset_path
        data = np.load(dataset_path)
        self.latents = torch.tensor(data["latents"], dtype=torch.float32)

        genera = data["genera"]
        gen_min, gen_max = np.min(genera), np.max(genera)
        # Shift to make sure labels start from 0
        shifted = genera - gen_min
        self.genera = torch.tensor(shifted, dtype=torch.long)
        self.num_classes = gen_max - gen_min + 1
        self.size = self.latents.shape[0]

    def __getitem__(self, idx):
        return self.latents[idx], self.genera[idx]

    def collate_fn(self, batch):
        latents, genera = zip(*batch)
        return torch.stack(latents), torch.stack(genera)

    def __len__(self):
        return self.size

""""
from scipy.interpolate import griddata
from matplotlib import pyplot as plt

x_new = np.linspace(min(xs), max(xs), 100)
y_new = np.linspace(min(ys), max(ys), 100)
X_grid, Y_grid = np.meshgrid(x_new, y_new)
Z_interpolated = griddata((xs, ys), volumes, (X_grid, Y_grid), method='linear')

plt.figure(figsize=(8, 6))
plt.contourf(X_grid, Y_grid, Z_interpolated, levels=20, cmap='viridis')
plt.colorbar(label='Interpolated Z value')
plt.scatter(xs, ys, c='red', s=10, label='Original Data Points')
plt.title('2D Interpolated Grid')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
"""