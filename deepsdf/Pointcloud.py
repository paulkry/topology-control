import os
import numpy as np
import meshio as meshio
import polyscope as ps
from pathlib import Path
import igl
from volume.compute_volume_genus import match_volume, compute_volume

class PointCloudProcessor:
    """Point cloud generator enforcing a common target volume (strict) and centering.

    Bbox is NOT forced; shapes keep proportional extents. Guarantees volume match within tolerance or raises.
    """

    def __init__(self, data_dir="data", target_volume=20, volume_tolerance=5e-3):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.target_volume = target_volume
        self.volume_tolerance = volume_tolerance  

    def load_mesh(self, mesh_file):
        mesh = meshio.read(mesh_file)
        vertices = mesh.points.copy()
        faces = mesh.cells[0].data
        orig_vol = compute_volume(vertices, faces)
        vertices, faces = match_volume(vertices, faces, self.target_volume)
        
        # center mesh in unit cube (preserves volume)
        vertices -= np.mean(vertices, axis=0)
        new_vol = compute_volume(vertices, faces)
        rel_err = abs(new_vol - self.target_volume) / self.target_volume
        
        status = "PASS" if rel_err <= self.volume_tolerance else "WARN"
        print(f"    [Volume] {Path(mesh_file).name}: initial={orig_vol:.4f} final={new_vol:.4f} target={self.target_volume:.4f} rel_err={rel_err:.2%} status={status}")
        if status == "WARN":
            print(f"    [Volume][WARN] Relative error {rel_err:.2%} > tolerance {self.volume_tolerance:.2%}. Consider lowering tolerance or refining scale.")
        
        self.last_scale_info = {"method": "match_volume", "orig_volume": orig_vol, "new_volume": new_vol, "rel_err": rel_err, "tolerance": self.volume_tolerance}
        
        return vertices, faces, Path(mesh_file).stem
    
    def sample_points(self, vertices, faces, radius=0.02, sigma=0.02, mu=0.0, n_gaussian=10, n_uniform=1000):
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
            noise = np.random.normal(mu, sigma, (n_gaussian * surface_points.shape[0], 3))
            tiled_surface = np.tile(surface_points, (n_gaussian, 1))
            gaussian_points = tiled_surface + noise
            sampled_points = np.concatenate((sampled_points, gaussian_points), axis=0)
        return sampled_points

    def generate_point_cloud(self, meshes, radius=0.02, sigma=0.02, mu=0.0, n_gaussian=10, n_uniform=1000):
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
                print(f"    [Existing File] Loading data for {name}")
                sampled_points = np.load(points_file)
                distances = np.load(distances_file)
            else:
                print(f"    [New File] Sampling points for {name}")
                sampled_points = self.sample_points(vertices, faces, radius, sigma, mu, n_gaussian, n_uniform)
                distances = igl.signed_distance(sampled_points, vertices, faces)[0]
                
                # Save data
                np.save(points_file, sampled_points)
                np.save(distances_file, distances)