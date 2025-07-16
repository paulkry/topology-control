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
        self.latents = torch.tensor(data["latents"], dtype=torch.float64)
        self.volumes = torch.tensor(data["volumes"], dtype=torch.float64)
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


            
class CDataProcessor:
    def __init__(self, config):
        """
        Initialize the data processor with configuration parameters.
        
        Parameters:
            config (dict): Configuration parameters for data processing including:
                - dataset_paths: dict with paths to raw data and processed data
                - point_cloud_params: dict with sampling parameters
                - mesh_files: list of mesh file names to process
        """
        self.config = config
        
        # Extract paths from config
        dataset_paths = config.get('dataset_paths', {})
        self.raw_data_path = dataset_paths.get('raw', 'data/raw')
        self.processed_data_path = dataset_paths.get('processed', 'data/processed')
        
        # Extract point cloud processing parameters
        pc_params = config.get('point_cloud_params', {})
        self.radius = pc_params.get('radius', 0.02)
        self.sigma = pc_params.get('sigma', 0.01)
        self.mu = pc_params.get('mu', 0.0)
        self.n_gaussian = pc_params.get('n_gaussian', 5)
        self.n_uniform = pc_params.get('n_uniform', 1000)
        
        # Get all mesh files from raw directory
        self.mesh_files = self._discover_mesh_files()
        
        # Initialize point cloud processor
        self.point_cloud_processor = PointCloudProcessor(data_dir=self.processed_data_path)
    
    def _discover_mesh_files(self):
        """
        Discover all mesh files in the raw data directory.
        
        Returns:
            list: List of mesh file names found in raw directory
        """
        if not os.path.exists(self.raw_data_path):
            print(f"Warning: Raw data path does not exist: {self.raw_data_path}")
            return []
        
        # Common mesh file extensions
        mesh_extensions = ['.obj', '.ply', '.stl', '.off', '.vtk']
        mesh_files = []
        
        for file in os.listdir(self.raw_data_path):
            file_path = os.path.join(self.raw_data_path, file)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file)
                if ext.lower() in mesh_extensions:
                    mesh_files.append(file)
        
        print(f"Discovered {len(mesh_files)} mesh files in {self.raw_data_path}: {mesh_files}")
        return mesh_files

    def process(self):
        """
        Main processing method that integrates with the pipeline.
        Uses the PointCloudProcessor to generate point clouds and signed distances.
        
        Returns:
            dict: Processing results including file paths and statistics
        """
        print(f"Processing {len(self.mesh_files)} mesh files...")
        
        # Build full paths to mesh files
        mesh_paths = []
        for mesh_file in self.mesh_files:
            mesh_path = os.path.join(self.raw_data_path, mesh_file)
            if os.path.exists(mesh_path):
                mesh_paths.append(mesh_path)
            else:
                print(f"Warning: Mesh file not found: {mesh_path}")
        
        if not mesh_paths:
            raise ValueError(f"No valid mesh files found in {self.raw_data_path}")
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Process each mesh file using the PointCloudProcessor approach
        processing_results = {
            'processed_files': [],
            'point_cloud_files': [],
            'signed_distance_files': [],
            'total_points_generated': 0,
            'processing_params': {
                'radius': self.radius,
                'sigma': self.sigma,
                'mu': self.mu,
                'n_gaussian': self.n_gaussian,
                'n_uniform': self.n_uniform
            }
        }
        
        for mesh_path in mesh_paths:
            result = self._process_single_mesh(mesh_path)
            processing_results['processed_files'].append(result['mesh_name'])
            processing_results['point_cloud_files'].append(result['points_file'])
            processing_results['signed_distance_files'].append(result['distances_file'])
            processing_results['total_points_generated'] += result['num_points']
        
        print(f"Data processing complete. Generated {processing_results['total_points_generated']} total points.")
        return processing_results
    
    def _process_single_mesh(self, mesh_path):
        """
        Process a single mesh file using the PointCloudProcessor logic.
        
        Parameters:
            mesh_path (str): Path to the mesh file
            
        Returns:
            dict: Results for this specific mesh
        """
        # Load and normalize mesh using PointCloudProcessor method
        vertices, faces, name = self.point_cloud_processor.load_mesh(mesh_path)
        
        # Define output file paths (following PointCloudProcessor naming convention)
        points_file = os.path.join(self.processed_data_path, f"{name}_sampled_points.npy")
        distances_file = os.path.join(self.processed_data_path, f"{name}_signed_distances.npy")
        
        # Check if data already exists (same logic as PointCloudProcessor)
        if os.path.exists(points_file) and os.path.exists(distances_file):
            print(f"  [Existing File] Loading data for {name}")
            sampled_points = np.load(points_file)
            distances = np.load(distances_file)
        else:
            print(f"  [New File] Sampling points for {name}")
            # Sample points using PointCloudProcessor method with config parameters
            sampled_points = self.point_cloud_processor.sample_points(
                vertices, faces, 
                radius=self.radius,
                sigma=self.sigma,
                mu=self.mu,
                n_gaussian=self.n_gaussian,
                n_uniform=self.n_uniform
            )
            
            # Compute signed distances (same as PointCloudProcessor)
            distances = igl.signed_distance(sampled_points, vertices, faces)[0]
            
            # Save data (same format as PointCloudProcessor)
            np.save(points_file, sampled_points)
            np.save(distances_file, distances)
            print(f"    Saved {len(sampled_points)} points to {points_file}")
            print(f"    Saved distances to {distances_file}")
        
        return {
            'mesh_name': name,
            'points_file': points_file,
            'distances_file': distances_file,
            'num_points': len(sampled_points),
            'vertices': vertices,
            'faces': faces
        }
    