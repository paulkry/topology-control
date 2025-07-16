# Latent Space Pathfinding (Experimental)

This folder contains minimal working code for testing our current idea:  
**learning to navigate latent space using the gradient of a guiding neural network.**

The setup is focused on experimenting with the idea of _path-finding in latent space_ guided by another model (e.g., a volume or topology regressor). We're exploring whether such guidance can help interpolate or transition meaningfully between latent codes.

## Core Components

- `model.py`: A neural regressor from latent vectors to volumes
- `config.py`: Customizable hyperparameters
- `dataset.py`: Dataset class
- `train.py`: Simple training pipeline with checkpoint saving
- `compute_volume.py`: utility functions for generating data



---

