import torch
import os
import meshio as meshio
from torch import optim
from torch.utils.data import DataLoader
from torch import nn 
import torch

from dataset import VolumeDataset
from model import Latent2Volume
from config import LATENT_DIM, DEV, BATCH_SIZE, LR, EPOCHS, VOLUME_DIR

if __name__ == "__main__":
    dataset = VolumeDataset(
        dataset_path=os.path.join(VOLUME_DIR, "data", "2d_latents_volumes.npz")  # Path to the dataset file,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)
    model = Latent2Volume(LATENT_DIM).to(DEV)
    crit = nn.L1Loss()
    opt = optim.Adam(list(model.parameters()), lr=LR)

    best_loss = float('inf')

    for e in range(EPOCHS):
        epoch_loss = 0.0
        for latents, volumes in loader:
            opt.zero_grad()
            latents, volumes = latents.to(DEV), volumes.to(DEV)
            predicted_volumes = model.forward(latents)
            loss = crit(predicted_volumes, volumes)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {e+1} | Avg Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss

            os.makedirs(os.path.join(VOLUME_DIR, "checkpoints"), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
            }, os.path.join(VOLUME_DIR, "checkpoints", "latent2volume_best.pt"))
            print(f"Best model saved at epoch {e+1} with loss {best_loss:.6f}")