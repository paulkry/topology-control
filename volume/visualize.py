import torch
import polyscope as ps
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
import os
import pyvista as pv

from compute_volume import get_volume_coords, generate_mesh_from_latent, predict_sdf
from sdfs import SDF_interpolator, sdf_2_torus, sdf_torus, sdf_sphere
from config import DEV, COORDS_FIRST, LATENT_FIRST, VOLUME_DIR, LATENT_VEC_MAX
import polyscope.imgui as psim

def create_interpolation_function(data_array):
    # converts array into continuous function between 0 and 1
    xp = np.linspace(0, 1, len(data_array))

    def interp_func(x):
        return torch.tensor([np.interp(x, xp, data_array[:, latent_dim]) for latent_dim in range(data_array.shape[1])], dtype=torch.float32)

    return interp_func

def visualize_sdf(sdf, latent=torch.tensor([0.1, 0.7]), type=COORDS_FIRST): 
    """
    Extract the SDF values and visualize as an isosurface using Polyscope volume grid.
    """
    coords, grid_size = get_volume_coords()
    
    sdf_values = predict_sdf(latent, coords, sdf, type)
    sdf_numpy = sdf_values.cpu().numpy().reshape((grid_size, grid_size, grid_size))
    
    bound_low = coords.min(dim=0)[0].cpu().numpy()
    bound_high = coords.max(dim=0)[0].cpu().numpy()
    
    ps.init()
    ps.set_up_dir("z_up")
    
    ps_grid = ps.register_volume_grid("sdf_grid", grid_size, bound_low, bound_high)
    
    ps_grid.add_scalar_quantity("sdf_values", sdf_numpy, defineenable_isosurface_viz=True, 
                               isosurface_level=0.0,  # Zero level set for SDF
                               isosurface_color=(0.2, 0.8, 0.2),
                               enable_gridcube_viz=False)  # Hide grid cubes
    
    ps.show()


import polyscope.imgui as psim

def visualize_interpolation_path(deepsdf, path, volume_regressor, type=COORDS_FIRST):
    curr_frame = 0
    
    coords, grid_size = get_volume_coords()
    path_function = create_interpolation_function(path.cpu().numpy())
    
    bound_low = coords.min(dim=0)[0].cpu().numpy()
    bound_high = coords.max(dim=0)[0].cpu().numpy()
    
    def myCallback():
        nonlocal curr_frame, latent, pred_volume, volume

        psim.TextUnformatted(f"Predicted volume: {pred_volume:.4f}")
        psim.TextUnformatted(f"Actual volume: {volume:.2f}")

        path_updated, curr_frame = psim.SliderFloat("Point in path", curr_frame, 0, 1)
        if path_updated:
            latent = path_function(curr_frame)

        latent_updateds = []
        for dim in range(path.shape[1]):
            latent_updated, latent_value = psim.SliderFloat(f"Latent dim {dim}", latent[dim].item(), -LATENT_VEC_MAX, LATENT_VEC_MAX)
            latent_updateds.append(latent_updated)
            latent[dim] = latent_value

        update_frame = path_updated or any(latent_updateds)
        
        if update_frame:
            pred_volume = volume_regressor(latent.unsqueeze(0).to(DEV)).view(-1).item()

            sdf_values = predict_sdf(latent, coords, deepsdf, type).flatten()
            V, F = generate_mesh_from_sdf(sdf_values, coords, grid_size)
            volume = compute_volume(V, F)

            sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size, grid_size))
            
            ps_grid.add_scalar_quantity(
                "sdf_values",
                sdf_values, 
                defined_on='nodes',
                enable_isosurface_viz=True, 
                isosurface_level=0.0,
                isosurface_color=(0.2, 0.8, 0.2),
                enable_gridcube_viz=False,
                enabled=True
            )
        
    ps.init()
    ps.set_up_dir("y_up")
    
    ps.set_automatically_compute_scene_extents(False)
    ps.set_length_scale(1)
    
    ps_grid = ps.register_volume_grid(
        "interpolation_grid",
        (grid_size, grid_size, grid_size),
        bound_low,
        bound_high
    )

    latent = path_function(0)
    pred_volume = volume_regressor(latent.unsqueeze(0).to(DEV)).view(-1).item()

    sdf_values = predict_sdf(latent, coords, deepsdf, type).flatten()
    V, F = generate_mesh_from_sdf(sdf_values, coords, grid_size)
    volume = compute_volume(V, F)

    sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size, grid_size))


    ps_grid.add_scalar_quantity(
        "sdf_values",
        sdf_values,
        defined_on='nodes',
        enable_isosurface_viz=True, 
        isosurface_level=0.0,
        isosurface_color=(0.2, 0.8, 0.2),
        enable_gridcube_viz=False,
        enabled=True
    )
    
    ps.set_user_callback(myCallback)
    ps.show()


def visualize_2d_path(model, path=[]):
    """
        Plot the path as a scatter plot on the model's latent space. 
        Also can be used to visualize the learned mapping from latent
        space to volume.
    """
    minix, miniy = path[:, 0].min(axis=0)[0], path[:, 1].min(axis=0)[0]
    maxix, maxiy = path[:, 0].max(axis=0)[0], path[:, 1].max(axis=0)[0]

    linx = np.linspace(minix, maxix, 100)
    liny = np.linspace(miniy, maxiy, 100)

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

def visualize_classifier_res(model):
    """
    Given a classifier model, plot a heatmap for its prediction
    on its 2d latent space.
    """
    lin = np.linspace(0, LATENT_VEC_MAX, 100)
    grid_x, grid_y = np.meshgrid(lin, lin)
    xs = grid_x.ravel()
    ys = grid_y.ravel()
    latents = torch.tensor(np.vstack((xs, ys)).T, dtype=torch.float32).to(DEV)

    with torch.no_grad():
        logits = model(latents)
        preds = logits.argmax(dim=1).detach().cpu().numpy()

    _visualize_heatmap(xs, ys, preds - 1)

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
    # sdf_interpolator = SDF_interpolator(sdf_sphere, sdf_torus, sdf_2_torus)

    # visualize_latent_vs_genera()
    # visualize_sdf(sdf_interpolator, latent=torch.tensor([0.29890096, 0.16535072]))

    # from compute_path import compute_path
    from compute_path_opt import compute_path
    from compute_volume import generate_mesh_from_sdf, compute_volume
    # # from compute_path_with_geodesic import compute_geodesic_path
    from model import Latent2Volume, Latent2Genera
    from config import LATENT_DIM, DEV

    checkpoint = torch.load("checkpoints/latent2volume_best_yuan2.pt", map_location=DEV)
    volume_regressor = Latent2Volume(LATENT_DIM).to(DEV)

    volume_regressor.load_state_dict(checkpoint["model_state_dict"])
    volume_regressor.eval()

    # visualize_2d_path(model)

    path = compute_path(
        torch.tensor([-2, 5], dtype=torch.float32).to(DEV),
        torch.tensor([3, 1.8], dtype=torch.float32).to(DEV),
        volume_regressor,
        20,
        smooth_term_w=0.001
    ).cpu()


    visualize_2d_path(volume_regressor, path=path)

    # Visualizing the 3d shapes from the paths
    model_path = os.path.join(VOLUME_DIR, "trained_deepsdfs", "sdfnet_model.pt")
    deepsdf = torch.jit.load(model_path).to(DEV)

    # path = [[0, 0], [0, 1], [1, 0], [0.4, 0], [0, 0.4], [0.29890096, 0.16535072]]
    # path = torch.tensor(path, dtype=torch.float32)

    visualize_interpolation_path(deepsdf, path, volume_regressor, LATENT_FIRST)
