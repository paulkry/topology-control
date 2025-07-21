import torch
import os
from enum import Enum

current_file_path = os.path.abspath(__file__) # Absolute path to the current file
VOLUME_DIR = os.path.dirname(current_file_path) # Directory containing the file
LATENT_DIM = 2
BATCH_SIZE = 32
LR = 1e-3  # scale depending on batch
DEV = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
LAYER_SIZE = 128
LATENT_VEC_MAX = 10 # what is the max entry of latent vec dims?

"""
whether model(latent, xyz) or model(xyz, latent). i.e., what the model expects
In particular, SDF_interpolator and current implementation of DeepSDF in src
belongs to the latter class, while the model provided by Yuanyuan is of the
former class.
"""
COORDS_FIRST = 1
LATENT_FIRST = 2
