import torch
import polyscope as ps
import igl
import numpy as np

from compute_volume import get_volume_coords
from sdfs import SDF_interpolator
from config import DEV


def visualize_sdf(sdf): 
    """
    Reconstruct the object from the latent code and visualize it.
    """
    coords, grid_size = get_volume_coords()
    sdf_values = sdf(torch.tensor([0.1, 0.7]), coords.to(DEV))

    vertices, faces, _ = igl.marching_cubes(np.array(sdf_values.cpu()), np.array(coords.cpu()), grid_size, grid_size, grid_size, 0.0)

    
    ps.init()
    ps_sdf = ps.register_surface_mesh("sdf visualization", vertices, faces)

    ps.show()

if __name__ == "__main__":
    sdf_interpolator = SDF_interpolator()
    visualize_sdf(sdf_interpolator)