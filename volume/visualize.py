import torch
import polyscope as ps
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
import os
import pyvista as pv

from compute_volume import get_volume_coords, generate_mesh
from sdfs import SDF_interpolator, sdf_2_torus, sdf_torus, sdf_sphere
from config import DEV, COORDS_FIRST, LATENT_FIRST, VOLUME_DIR, LATENT_VEC_MAX
import polyscope.imgui as psim


def visualize_sdf(sdf, latent = torch.tensor([0.1, 0.7]), type=COORDS_FIRST): 
    """
    Extract the mesh from an interpolated sdf/deepsdf and visualize it.
    """
    coords, grid_size = get_volume_coords()
    vertices, faces = generate_mesh(latent, coords, grid_size, sdf, type)

    ps.init()
    ps_sdf = ps.register_surface_mesh("sdf visualization", vertices, faces)

    ps.show()

def visualize_interpolation_path(model, path, type=COORDS_FIRST):
    n_timestep = len(path)
    curr_frame = 0
    auto_playing = True

    all_vertices = []
    all_faces = []
    coords, grid_size = get_volume_coords()

    for i, latent in enumerate(path):
        V, F = generate_mesh(latent, coords, grid_size, model, type)
        all_vertices.append(V)
        all_faces.append(F)

    def myCallback():
        nonlocal curr_frame, auto_playing

        update_frame = False
        _, auto_playing = psim.Checkbox("Autoplay", auto_playing)

        if auto_playing:
            update_frame = True
            curr_frame = (curr_frame + 1) % n_timestep

        slider_updated, curr_frame = psim.SliderInt("Current Shape", curr_frame, 0, n_timestep-1)
        update_frame = update_frame or slider_updated

        if update_frame:
            if ps.has_surface_mesh("interpolated_shape"):
                ps.remove_surface_mesh("interpolated_shape")
            ps.register_surface_mesh("interpolated_shape", all_vertices[curr_frame], all_faces[curr_frame])

    ps.init()
    ps.set_up_dir("z_up")

    ps.set_automatically_compute_scene_extents(False)
    ps.set_length_scale(1)

    ps.set_user_callback(myCallback)
    ps.register_surface_mesh("interpolated_shape", all_vertices[0], all_faces[0])
    ps.show()


def visualize_2d_path(model, path=[]):
    """
        Plot the path as a scatter plot on the model's latent space. 
        Also can be used to visualize the learned mapping from latent
        space to volume.
    """
    minix, miniy = path[:, 0].min(axis=0)[0], path[:, 1].min(axis=0)[0]
    maxix, maxiy = path[:, 0].max(axis=0)[0], path[:, 1].max(axis=0)[0]

    linx = np.linspace(minix-10, maxix+10, 100)
    liny = np.linspace(miniy-10, maxiy+10, 100)

    grid_x, grid_y = np.meshgrid(linx, liny)
    xs = grid_x.ravel()
    ys = grid_y.ravel()
    latents = torch.tensor(np.vstack((xs, ys)).T, dtype=torch.float32).to(DEV)

    predicted_volumes = model(latents).detach().cpu().numpy()

    # if path is empty just visualize the heatmap
    if path is None:
        _visualize_heatmap(xs, ys, predicted_volumes)
        return

    path = path.cpu().numpy()
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

def visualize_latent_vs_genera(path=os.path.join(VOLUME_DIR, "data", "2d_latents_volumes.npz")):
    """
        Plot 2d latent vs genera
    """
    data = np.load(path)
    latents = data["latents"]
    genera = data["genera"]

    idx = np.where(genera == 3)[0][0]
    print(latents[idx])

    plt.figure(figsize=(6, 6))
    cmap = ListedColormap(plt.get_cmap('tab10').colors[:5])
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=genera, cmap=cmap, s=5)

    # Optional: colorbar with label names
    cbar = plt.colorbar(scatter, ticks=range(np.min(genera), np.max(genera)+1))
    cbar.set_label('Genera')
    cbar.set_ticks([0, 1, 2, 3, 4])

    plt.xlabel('Latent dim 1')
    plt.ylabel('Latent dim 2')
    plt.title('Latent vs. genera')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

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
    sdf_interpolator = SDF_interpolator(sdf_sphere, sdf_torus, sdf_2_torus)
    # visualize_latent_vs_genera()
    # visualize_sdf(sdf_interpolator, latent=torch.tensor([0.29890096, 0.16535072]))

    # from compute_path import compute_path, compute_path2
    from compute_path_opt import compute_path
    # # from compute_path_with_geodesic import compute_geodesic_path
    from model import Latent2Volume
    from config import LATENT_DIM, DEV

    checkpoint = torch.load("checkpoints/latent2volume_best.pt", map_location=DEV)
    model = Latent2Volume(LATENT_DIM).to(DEV)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # visualize_2d_path(model)

    path = compute_path(torch.tensor([0, 10], dtype=torch.float32).to(DEV), torch.tensor([10, 10], dtype=torch.float32).to(DEV), model, 20, smooth_term_w=0.01).cpu()

    print("Path found with length:", len(path))

    visualize_2d_path(model, path=path)

    # Visualizing the 3d shapes from the paths
    # model_path = os.path.join(VOLUME_DIR, "trained_deepsdfs", "sdfnet_model.pt")
    # deepsdf = torch.jit.load(model_path).to(DEV)

    # path = [[0, 0], [0, 1], [1, 0], [0.4, 0], [0, 0.4], [0.29890096, 0.16535072]]
    # path = torch.tensor(path, dtype=torch.float32)

    visualize_interpolation_path(sdf_interpolator, path, COORDS_FIRST)
