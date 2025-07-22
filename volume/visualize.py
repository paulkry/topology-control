import torch
import polyscope as ps
import igl
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import os

from compute_volume import get_volume_coords, predict_sdf
from sdfs import SDF_interpolator
from config import DEV, COORDS_FIRST, LATENT_FIRST, VOLUME_DIR, LATENT_VEC_MAX


def visualize_sdf(sdf, latent = torch.tensor([0.1, 0.7]), type=COORDS_FIRST): 
    """
    Extract the mesh from an interpolated sdf/deepsdf and visualize it.
    """
    coords, grid_size = get_volume_coords()
    sdf_values = predict_sdf(latent, coords, sdf, type).flatten()


    vertices, faces, _ = igl.marching_cubes(sdf_values, coords.cpu().numpy(), grid_size, grid_size, grid_size, 0.0)

    
    ps.init()
    ps_sdf = ps.register_surface_mesh("sdf visualization", vertices, faces)

    ps.show()

def visualize_interpolation_path(model, path, type=COORDS_FIRST):
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
        V, F = construct_from_latent(model, latent, "helix", i, type)
        ps.register_surface_mesh(f"path_{i+1}", V, F)
    ps.show()

def construct_from_latent(model, latent, positioning, index, type):
    """
    Reconstruct the mesh from a latent vector using the model
    
    Parameters:
        model: Trained DeepSDF model or SDF_interpolator object
        latent: A latent vector
        positioning: helix (circular) or row
        index: the index used for angle or shift calculation
    Returns:
        vertices, faces
    """
    coords, grid_size = get_volume_coords()
    sdf = predict_sdf(latent.to(DEV), coords, model, type).flatten()

    vertices, faces, _ = igl.marching_cubes(
        sdf.detach().cpu().numpy(), 
        coords.cpu().numpy(), 
        grid_size, grid_size, grid_size, 0.0
    )
    if positioning == "row":
        vertices[:, 0] += 2 * index
    else:
        angle = (index % 10) * 2 * np.pi / 10 
        vertices[:, 0] += 3 * np.cos(angle)
        vertices[:, 1] += 3 * np.sin(angle)
        vertices[:, 2] += 0.2 * index
    
    return vertices, faces

def visualize_2d_path(model, path=[]):
    """
        Plot the path as a scatter plot on the model's latent space. 
        Also can be used to visualize the learned mapping from latent
        space to volume.
    """
    lin = np.linspace(0, LATENT_VEC_MAX, 100)
    grid_x, grid_y = np.meshgrid(lin, lin)
    xs = grid_x.ravel()
    ys = grid_y.ravel()
    latents = torch.tensor(np.vstack((xs, ys)).T, dtype=torch.float32).to(DEV)

    predicted_volumes = model(latents).detach().cpu().numpy()

    # if path is empty just visualize the heatmap
    if path is None:
        _visualize_heatmap(xs, ys, predicted_volumes)
        return

    path = np.array(path)
    path_xs = path[:, 0]
    path_ys = path[:, 1]

    _visualize_heatmap(xs, ys, predicted_volumes, path_xs, path_ys)


def visualize_latent_vs_volume(path=os.path.join(VOLUME_DIR, "data", "2d_latents_volumes.npz")):
    """
        Plot the 2d latent vs volume data as a heat map
    """
    data = np.load(path)
    latents = data["latents"]
    volumes = data["volumes"]

    xs, ys = latents[:, 0], latents[:, 1]

    _visualize_heatmap(xs, ys, volumes)

def _visualize_heatmap(X, Y, Z, pointsX = None, pointsY = None):
    """
    Visualize Z on XY-plane as a heatmap
        Parameters:
        (X, Y): Data points
        Z: Values used for generating heatmap
        (pointsX, pointsY): Points to scatter plot
    """
    x_new = np.linspace(min(X), max(X), 1000)
    y_new = np.linspace(min(Y), max(Y), 1000)
    X_grid, Y_grid = np.meshgrid(x_new, y_new)
    Z_interpolated = griddata((X, Y), Z, (X_grid, Y_grid), method='linear')

    plt.figure(figsize=(8, 6))
    plt.contourf(X_grid, Y_grid, Z_interpolated, levels=20, cmap='viridis')
    plt.colorbar(label='Interpolated Z value')
    if pointsX is not None and pointsY is not None:
        plt.scatter(pointsX, pointsY, c='red', s=10, label='Original Data Points')

    for i, (x, y) in enumerate(zip(pointsX, pointsY)):
        plt.annotate(str(i), (x,y), fontsize=8, color='black')

    plt.title('2D Interpolated Grid')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    from compute_path_opt import compute_path
    # from compute_path_with_geodesic import compute_geodesic_path
    from model import Latent2Volume
    from config import LATENT_DIM, DEV

    checkpoint = torch.load("checkpoints/latent2volume_best.pt", map_location=DEV)
    model = Latent2Volume(LATENT_DIM).to(DEV)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    path = compute_path(
        torch.tensor([2, 0], dtype=torch.float32).to(DEV),
        torch.tensor([1, 4], dtype=torch.float32).to(DEV),
        model,
        10,
        smooth_term_w=1e-5
    ).cpu()

    # # print("Path found with length:", len(path))
    # path = [x.detach() for x in path]
    visualize_2d_path(model, path=path)

    # Visualizing the 3d shapes from the paths
    # model_path = os.path.join(VOLUME_DIR, "trained_deepsdfs", "sdfnet_model.pt")
    # deepsdf = torch.jit.load(model_path).to(DEV)

    # path = [[0, 0], [10, 100]]
    # path = torch.tensor(path)

    visualize_interpolation_path(model, path, LATENT_FIRST)