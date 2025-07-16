"""
Data Processing for Topology Control
"""
import torch
import os
import numpy as np
import meshio as meshio
import igl
import polyscope as ps
from pathlib import Path
from torch.utils.data import Dataset

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
        
class CDataProcessor:
    def __init__(self, config):
        """
        Initialize the data processor with configuration parameters.
        
        Parameters:
            config (dict): Configuration parameters for data processing.
        """
        self.config = config

    def process_data(self, raw_data):
        """
        Process the raw data according to the configuration.
        
        Parameters:
            raw_data: The input data to be processed.
        
        Returns:
            Processed data.
        """
        # Implement data processing logic here
        return raw_data  # Placeholder for processed data
    
"""
Dataset Class for training network
mapping latent vectors to volumes
"""
class VolumeDataset(Dataset):

    def __init__(self, dataset_path):
        """
            Expecting a npz file with two arrays:
            latents and volumes
        """
        self.file_path = dataset_path
        data = np.load(dataset_path)
        self.latents = torch.tensor(data["latents"], dtype=torch.float32)
        self.volumes = torch.tensor(data["volumes"], dtype=torch.float32)
        self.size = self.latents.shape[0]

    def __getitem__(self, idx):
        return self.latents[idx], self.volumes[idx]

    def collate_fn(self, batch):
        latents, volumes = zip(*batch)
        return torch.stack(latents), torch.stack(volumes)

    def __len__(self):
        return self.size


def main():
    """Example usage (of PointCloudProcessor)"""
    # Get mesh paths (src and data are on the same level)
    project_root = Path(__file__).parent.parent
    data_dir = project_root/"data"
    
    meshes = [
        str(data_dir/"raw"/"bunny.obj"),
        str(data_dir/"raw"/"bimba.obj"),
        str(data_dir/"raw"/"torus.obj")
    ]
    
    # Generate point clouds
    processor = PointCloudProcessor(data_dir=str(data_dir))
    processor.generate_point_cloud(meshes, sigma=0.01, n_gaussian=5)


if __name__ == "__main__":
    main()