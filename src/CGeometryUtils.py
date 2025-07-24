import os
import torch
import numpy as np
import meshio as meshio
# import polyscope as ps
from pathlib import Path
import igl
from tqdm import tqdm
import polyscope as ps
"""
   Functions for visualization and processing of meshes
"""

class MeshRenderer:
    """Dedicated class for mesh visualization"""
    
    def __init__(self, renderer='polyscope'):
        self.renderer = renderer
        if renderer == 'polyscope':
            ps.init()
    
    def render_mesh(self, vertices, faces, name="mesh"):
        """Render a single mesh"""
        if self.renderer == 'polyscope':
            ps.register_surface_mesh(name, vertices, faces)
            return ps.get_surface_mesh(name)
        else:
            raise ValueError(f"Renderer {self.renderer} not supported")
    
    def render_point_cloud(self, points, name="points", sdf_values=None):
        """Render point cloud with optional SDF coloring"""
        if self.renderer == 'polyscope':
            pc = ps.register_point_cloud(name, points)
            if sdf_values is not None:
                pc.add_scalar_quantity("signed_distance", sdf_values)
            return pc
        else:
            raise ValueError(f"Renderer {self.renderer} not supported")
    
    def render_mesh_with_points(self, vertices, faces, points, sdf_values=None, mesh_name="mesh", points_name="points"):
        """Render mesh and point cloud together"""
        self.render_mesh(vertices, faces, mesh_name)
        self.render_point_cloud(points, points_name, sdf_values)
    
    def show(self):
        """Display the visualization"""
        if self.renderer == 'polyscope':
            ps.show()
    
    def clear(self):
        """Clear all visualizations"""
        if self.renderer == 'polyscope':
            ps.remove_all_structures()
    
class PointCloudProcessor:
    """Pipeline for generating point clouds from meshes"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # self.renderer = MeshRenderer()
    
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
        print('compute blue noise')
        surface_points = igl.blue_noise(vertices, faces, radius)[2]
        # Concatenate surface points with random points
        print('concat')
        sampled_points = np.concatenate((random_points, surface_points), axis=0)
        
        # Add Gaussian noise if needed
        if n_gaussian > 0 and sigma > 0:
            noise = np.random.normal(mu, sigma, (n_gaussian, surface_points.shape[0], 3))
            print("entering tqdm")
            for i in tqdm(range(n_gaussian)):
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
            # self.renderer.render_mesh_with_points(vertices, faces, sampled_points, distances, name, f"{name}_points")
        
        # self.renderer.show()
    
    def get_dataset_stats(self, mesh_name):
        """Get statistics about generated dataset"""
        points_file = f"{self.data_dir}/{mesh_name}_sampled_points.npy"
        distances_file = f"{self.data_dir}/{mesh_name}_signed_distances.npy"
        
        if os.path.exists(points_file) and os.path.exists(distances_file):
            points = np.load(points_file)
            distances = np.load(distances_file)
            
            print(f"Dataset stats for {mesh_name}:")
            print(f"  Total points: {len(points)}")
            print(f"  SDF range: [{distances.min():.4f}, {distances.max():.4f}]")
            print(f"  Inside surface (SDF < 0): {(distances < 0).sum()}")
            print(f"  Outside surface (SDF > 0): {(distances > 0).sum()}")
            print(f"  Near surface (|SDF| < 0.01): {(np.abs(distances) < 0.01).sum()}")
            
            return {'points': points, 'distances': distances}
        else:
            print(f"No dataset found for {mesh_name}")
            return None
    
class VolumeProcessor:
    """Dedicated processor for volume sampling and grid operations"""
    
    def __init__(self, device="cpu", resolution=50):
        """
        Initialize volume processor
        
        Parameters:
            device (str): Device for volume coordinates computation
            resolution (int): Volume grid resolution
        """
        self.device = device
        self.resolution = resolution
        
        # Initialize volume coordinates for sampling
        self.volume_coords, self.grid_size_axis = self._get_volume_coords(device, resolution)
        
        print(f"Volume Processor initialized with {resolution}³ volume grid on {device}")
    
    def _get_volume_coords(self, device, resolution=50):
        """Get 3-dimensional vector (M, N, P) according to the desired resolutions."""
        # FIX: Use linspace to get exactly 'resolution' points from -1 to 1
        grid_values = torch.linspace(-1, 1, resolution).to(device)
        grid = torch.meshgrid(grid_values, grid_values, grid_values, indexing='ij')  # 3D grid
        grid_size_axis = grid_values.shape[0]
        coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)
        return coords, grid_size_axis
    
    def sample_volume_points(self, num_samples=1000):
        """
        Sample random points from the volume grid
        
        Parameters:
            num_samples (int): Number of volume samples
            
        Returns:
            np.ndarray: Random volume points
        """
        random_indices = np.random.choice(
            range(self.volume_coords.shape[0]), num_samples, replace=False
        )
        return self.volume_coords[random_indices].cpu().numpy()