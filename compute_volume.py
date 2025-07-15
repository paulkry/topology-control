import igl
import polyscope as ps
import numpy as np
import meshio as meshio

import numpy as np
import trimesh

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_volume_coords(resolution = 50):
    """Get 3-dimensional vector (M, N, P) according to the desired resolutions."""
    # Define grid
    grid_values = torch.arange(-1, 1, float(1/resolution)).to(device) # e.g. 50 resolution -> 1/50 
    grid = torch.meshgrid(grid_values, grid_values, grid_values)
    
    grid_size_axis = grid_values.shape[0]

    # Reshape grid to (M*N*P, 3)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)

    return coords, grid_size_axis


def predict_sdf(latent, coords, model):

    # return np.linalg.norm(coords, axis=1) - 1

    sdf = torch.tensor([], dtype=torch.float32).view(1, 0).to(device)

    latent_batch = latent.unsqueeze(0)  
    # Split coords into batches because of memory limitations
    coords_batches = torch.split(coords, 100000)

    model.eval()
    with torch.no_grad():
        for coords in coords_batches:
            coords = coords.unsqueeze(0)
            sdf_batch = model(latent_batch, coords)
            sdf = torch.hstack((sdf, sdf_batch.view(1, -1)))        

    return sdf


def compute_volume(latent_code, model, grid_size): 
    """
    Reconstruct the object from the latent code and visualize it.
    """
    coords, grid_size_axis = get_volume_coords(50)
    sdf = predict_sdf(latent_code, coords, model).flatten()

    vertices, faces, _ = igl.marching_cubes(np.array(sdf.to("cpu")), np.array(coords.to("cpu")), grid_size, grid_size, grid_size, 0.0)

    volume = triangle_mesh_to_volume(vertices, faces)
    
    return volume


def is_mesh_closed(vertices, faces):
    """
    Check if mesh is closed (watertight)
    """
    edges = set()
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            if edge in edges:
                edges.remove(edge)
            else:
                edges.add(edge)
    return len(edges) == 0

def triangle_mesh_to_volume(vertices, faces):
    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)
    if not isinstance(faces, np.ndarray):
        faces = np.array(faces)

    is_closed = is_mesh_closed(vertices, faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    volume = abs(mesh.volume)
        
    return volume, is_closed

# Example usage
if __name__ == "__main__":
    # Example 1: Create a simple cube mesh
    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]
    ])
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
    ])
    
    # Calculate volume
    volume, is_closed = triangle_mesh_to_volume(vertices, faces)
    
    print(f"Volume: {volume:.6f}")
    print(f"Is closed: {is_closed}")
    print(f"Expected volume (unit cube): 1.0")
    
    # Example 2: Load from OBJ file (uncomment if you have an OBJ file)
    # vertices, faces = load_obj_file("your_mesh.obj")
    # result = triangle_mesh_to_volume(vertices, faces, method='pytetgen')
    # save_results(result, "volume_results.txt")