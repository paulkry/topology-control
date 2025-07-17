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