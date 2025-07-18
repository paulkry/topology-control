import torch
import polyscope as ps
import igl
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

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
    ps.set_up_dir("z_up")
    for i, latent in enumerate(path):
        V, F = construct_from_latent(model, latent, "helix", 2 * i * np.pi / len(path))
        ps.register_surface_mesh(f"path_{i+1}", V, F)
    ps.show()

def visualize_2d_path(model, path):
    lin = np.linspace(0, 1, 100)
    grid_x, grid_y = np.meshgrid(lin, lin)
    xs = grid_x.ravel()
    ys = grid_y.ravel()
    latents = torch.tensor(np.vstack((xs, ys)).T, dtype=torch.float32).to(DEV)

    predicted_volumes = model(latents).detach().cpu().numpy()

    x_new = np.linspace(min(xs), max(xs), 100)
    y_new = np.linspace(min(ys), max(ys), 100)
    X_grid, Y_grid = np.meshgrid(x_new, y_new)
    Z_interpolated = griddata((xs, ys), predicted_volumes, (X_grid, Y_grid), method='linear')

    path = np.array(path)
    path_xs = path[:, 0]
    path_ys = path[:, 1]

    plt.figure(figsize=(8, 6))
    plt.contourf(X_grid, Y_grid, Z_interpolated, levels=20, cmap='viridis')
    plt.colorbar(label='Interpolated Z value')
    plt.scatter(path_xs, path_ys, c='red', s=10, label='Path')
    plt.title('2D Interpolated Grid')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

def construct_from_latent(model, latent, positioning, parameter):
    """
    Reconstruct the mesh from a latent vector using the model
    
    Parameters:
        model: Trained DeepSDF model or SDF_interpolator object
        latent: A latent vector
        positioning: helix (circular) or row
        parameter: angle or shift
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
    if positioning == "row":
        vertices[:, 0] += parameter
    else:
        vertices[:, 0] += 3 * np.cos(parameter)
        vertices[:, 1] += 3 * np.sin(parameter)
        vertices[:, 2] += 0.2 * parameter
    
    return vertices, faces


if __name__ == "__main__":
    sdf_interpolator = SDF_interpolator()
    # visualize_sdf(sdf_interpolator)

    from compute_path import compute_path, compute_path2
    from model import Latent2Volume
    from config import LATENT_DIM, DEV

    checkpoint = torch.load("checkpoints/latent2volume_best.pt", map_location=DEV)
    model = Latent2Volume(LATENT_DIM).to(DEV)

    # Load the saved state_dict (weights)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    path = compute_path2(torch.tensor([0, 1], dtype=torch.float32), torch.tensor([1, 0], dtype=torch.float32), model, 30)

    print("Path found with length:", len(path))
    # print(path)
    path = [x.detach() for x in path]

    visualize_interpolation_path(sdf_interpolator, path)
    visualize_2d_path(model, path)