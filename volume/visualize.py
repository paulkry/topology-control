import torch
import polyscope as ps
import igl
import numpy as np

from compute_volume import get_volume_coords, predict_sdf
from sdfs import SDF_interpolator
from config import DEV


def visualize_sdf(sdf): 
    """
    Extract the mesh from an interpolated sdf and visualize it.
    """
    coords, grid_size = get_volume_coords()
    sdf_values = sdf(torch.tensor([0.1, 0.7]), coords.to(DEV))

    vertices, faces, _ = igl.marching_cubes(np.array(sdf_values.cpu()), np.array(coords.cpu()), grid_size, grid_size, grid_size, 0.0)

    
    ps.init()
    ps_sdf = ps.register_surface_mesh("sdf visualization", vertices, faces)

    ps.show()

def visualize_interpolation_path(model, path):
    """
    Visualizing the meshes generated from each latent along the path by rendering them in a row.
    Each shape is placed 2 units apart from the previous one to avoid overlap.

    Parameters:
        model: Trained DeepSDF model or SDF_interpolator object
        path: List of latent vectors (torch.Tensor), e.g., from latent_A to latent_B
    """

    ps.init()
    for i, latent in enumerate(path):
        V, F = construct_from_latent(model, latent, 2 * i)
        ps.register_surface_mesh(f"path_{i+1}", V, F)
    ps.show()

def construct_from_latent(model, latent, shift):
    """
    Reconstruct the mesh from a latent vector using the model
    
    Parameters:
        model: Trained DeepSDF model or SDF_interpolator object
        latent: A latent vector
        shift: Shift the mesh by how much?
    Returns:
        vertices, faces
    """
    coords, grid_size = get_volume_coords()
    sdf = predict_sdf(latent.to(DEV), coords, model).flatten()

    vertices, faces, _ = igl.marching_cubes(
        sdf.cpu().numpy(), 
        coords.cpu().numpy(), 
        grid_size, grid_size, grid_size, 0.0
    )

    vertices[:, 0] += shift
    
    return vertices, faces


if __name__ == "__main__":
    sdf_interpolator = SDF_interpolator()
    visualize_sdf(sdf_interpolator)