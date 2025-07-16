import os
import numpy as np
import meshio as meshio
import igl
import polyscope as ps
from pathlib import Path

"""
   [To be Completed] Functions for visualization and processing of meshes
"""

class Renderer:
    def __init__(self, mesh):
        """
        Parameters:
            To be Defined
        """
        self.mesh = mesh

    def render(self):
        return None

    def save(self, filename):
        return None
    
class PointCloudProcessor:
    """Pipeline for generating point clouds from meshes"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def load_mesh(self, mesh_file):
        """Load and normalize mesh"""
        mesh = meshio.read(mesh_file)
        vertices = mesh.points.copy()
        # Rescale vertices to fit in the -1 to 1 cube
        vertices -= np.mean(vertices, axis=0)	
        vertices /= np.max(np.abs(vertices))
        faces = mesh.cells[0].data
        name = Path(mesh_file).stem
        return vertices, faces, name
    
    def sample_points(self, vertices, faces, radius=0.02, sigma=0.0, mu=0.0, n_gaussian=10, n_uniform=1000):
        """ 
        Sample points using different strategies
        Input: 
            - vertices: Mesh vertices
            - faces: Mesh faces
            - radius: Radius for blue noise sampling
            - sigma: Standard deviation for Gaussian noise
            - mu: Mean for Gaussian noise
            - n_gaussian: Number of sampled points to add Gaussian Noise
            - n_uniform: Number of uniform random samples
        
        Output:
            - sampled_points: Array of sampled points
        """
        # Sample random points in the -1 to 1 cube
        random_points = np.random.uniform(-1, 1, (n_uniform, 3))
        # Compute surface points using blue noise
        surface_points = igl.blue_noise(vertices, faces, radius)[2]
        # Concatenate surface points with random points
        sampled_points = np.concatenate((random_points, surface_points), axis=0)
        
        # Add Gaussian noise if needed
        if n_gaussian > 0 and sigma > 0:
            noise = np.random.normal(mu, sigma, (n_gaussian, surface_points.shape[0], 3))
            for i in range(n_gaussian):
                new_points = surface_points + noise[i]
                sampled_points = np.concatenate((sampled_points, new_points), axis=0)
        
        return sampled_points

    def generate_point_cloud(self, meshes, radius=0.02, sigma=0.0, mu=0.0, n_gaussian=10, n_uniform=1000):
        """
        Generate point clouds from meshes
            Input:
                - meshes: List of mesh file paths
                - radius: Radius for blue noise sampling
                - sigma: Standard deviation for Gaussian noise
                - mu: Mean for Gaussian noise
                - n_gaussian: Number of Gaussian noise samples
                - n_uniform: Number of uniform random samples
                
            Output:
                - Point cloud visualization using Polyscope
        """
        ps.init()
        
        for mesh_file in meshes:
            vertices, faces, name = self.load_mesh(mesh_file)
            
            # Check if data already exists
            points_file = f"{self.data_dir}/{name}_sampled_points.npy"
            distances_file = f"{self.data_dir}/{name}_signed_distances.npy"
            
            # Load data
            if os.path.exists(points_file) and os.path.exists(distances_file):
                print(f"[Existing File] Loading data for {name}")
                sampled_points = np.load(points_file)
                distances = np.load(distances_file)
            else:
                print(f"[New File] Sampling points for {name}")
                sampled_points = self.sample_points(vertices, faces, radius, sigma, mu, n_gaussian, n_uniform)
                distances = igl.signed_distance(sampled_points, vertices, faces)[0]
                
                # Save data
                np.save(points_file, sampled_points)
                np.save(distances_file, distances)
            
            # Visualize on Polyscope
            ps.register_surface_mesh(name, vertices, faces)
            point_cloud = ps.register_point_cloud(f"{name}_points", sampled_points)
            point_cloud.add_scalar_quantity("signed_distance", distances)
        
        ps.show()
      