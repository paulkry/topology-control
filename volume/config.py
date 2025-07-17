import torch
import os

current_file_path = os.path.abspath(__file__) # Absolute path to the current file
VOLUME_DIR = os.path.dirname(current_file_path) # Directory containing the file
LATENT_DIM = 2
BATCH_SIZE = 8
LR = 1e-3  # scale depending on batch
DEV = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
LAYER_SIZE = 128