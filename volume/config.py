import torch

LATENT_DIM = 128
BATCH_SIZE = 8  # maybe reduce for CPU
LR = 1e-3  # scale depending on batch
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
LAYER_SIZE = 128