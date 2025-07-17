"""
Data Processing for Topology Control
"""
import os
import numpy as np
import meshio as meshio
import igl
from src.CGeometryUtils import PointCloudProcessor, VolumeProcessor
  
class CDataProcessor:
    def __init__(self, config):
        """
        Initialize the data processor with configuration parameters.
        
        Parameters:
            config (dict): Configuration parameters for data processing including:
                - dataset_paths: dict with 'raw' and 'processed' paths (train/val auto-derived)
                - point_cloud_params: dict with sampling parameters
                - train_val_split: float between 0 and 1 for train/validation split ratio
        """
        self.config = config
        
        # Extract paths from config
        dataset_paths = config.get('dataset_paths', {})
        raw_path = dataset_paths.get('raw', 'data/raw')
        processed_path = dataset_paths.get('processed', 'data/processed')
        
         # Handle relative paths - make them relative to the project root
        if not os.path.isabs(raw_path):
            # Get the project root (parent of src directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)  # Go up from src to project root
            self.raw_data_path = os.path.join(project_root, raw_path)
            self.processed_data_path = os.path.join(project_root, processed_path)
        else:
            self.raw_data_path = raw_path
            self.processed_data_path = processed_path
            
        # Automatically derive train and val paths from processed path
        self.train_data_path = os.path.join(self.processed_data_path, 'train')
        self.val_data_path = os.path.join(self.processed_data_path, 'val')
        
        # Extract train/val split ratio
        self.train_val_split = config.get('train_val_split', 0.8)  # Default 80% train, 20% val
        
        # Extract point cloud processing parameters
        pc_params = config.get('point_cloud_params', {})
        self.radius = pc_params.get('radius', 0.02)
        self.sigma = pc_params.get('sigma', 0.01)
        self.mu = pc_params.get('mu', 0.0)
        self.n_gaussian = pc_params.get('n_gaussian', 5)
        self.n_uniform = pc_params.get('n_uniform', 1000)
        
        # Extract the volume processor parameter
        v_params = config.get('volume_processor_params', {})
        self.device = v_params.get('device', 'cpu')  # Default to CPU
        self.resolution = v_params.get('resolution', 64)  # Default resolution
        
        # Get all mesh files from raw directory
        self.mesh_files = self._discover_mesh_files()
        
        # Initialize point cloud processor
        self.point_cloud_processor = PointCloudProcessor(data_dir=self.processed_data_path)
        
        # Initialize volume processor 
        self.volume_processor = VolumeProcessor(device=self.device, resolution=self.resolution)
    
    def process(self):
        """
        Main processing method that integrates with the pipeline.
        Uses the PointCloudProcessor to generate point clouds and signed distances.
        Splits processed files into train and val directories.
        
        Returns:
            dict: Processing results including file paths and basic data statistics
        """
        print(f"Processing {len(self.mesh_files)} mesh files...")
        
        # Validate train_val_split
        if not 0 < self.train_val_split < 1:
            raise ValueError(f"train_val_split must be between 0 and 1, got {self.train_val_split}")
        
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
        
        # Create all necessary directories
        for path in [self.processed_data_path, self.train_data_path, self.val_data_path]:
            os.makedirs(path, exist_ok=True)
        
        # Determine train/val split
        train_files, val_files = self._split_files(mesh_paths)
        
        # Process files and organize into train/val
        processing_results = {
            'processed_files': [],
            'train_files': [],
            'val_files': [],
            'point_cloud_files': {'train': [], 'val': []},
            'signed_distance_files': {'train': [], 'val': []},
            'total_points_generated': 0,
            'train_count': len(train_files),
            'val_count': len(val_files),
            'split_ratio': self.train_val_split,
            'processing_params': {
                'radius': self.radius,
                'sigma': self.sigma,
                'mu': self.mu,
                'n_gaussian': self.n_gaussian,
                'n_uniform': self.n_uniform
            }
        }
        
            
        # Process training files
        print(f"Processing {len(train_files)} training files...")
        for mesh_path in train_files:
            result = self._process_single_mesh(mesh_path, 'train')
            if result is not None:  # Only add if processing succeeded
                processing_results['processed_files'].append(result['mesh_name'])
                processing_results['train_files'].append(result['mesh_name'])
                processing_results['point_cloud_files']['train'].append(result['points_file'])
                processing_results['signed_distance_files']['train'].append(result['distances_file'])
                processing_results['total_points_generated'] += result['num_points']
        
        # Process validation files
        print(f"Processing {len(val_files)} validation files...")
        for mesh_path in val_files:
            result = self._process_single_mesh(mesh_path, 'val')
            if result is not None:  # Only add if processing succeeded
                processing_results['processed_files'].append(result['mesh_name'])
                processing_results['val_files'].append(result['mesh_name'])
                processing_results['point_cloud_files']['val'].append(result['points_file'])
                processing_results['signed_distance_files']['val'].append(result['distances_file'])
                processing_results['total_points_generated'] += result['num_points']
        
        # Update counts to reflect actual processed files
        processing_results['train_count'] = len(processing_results['train_files'])
        processing_results['val_count'] = len(processing_results['val_files'])
        
        print(f"Data processing complete.")
        print(f"  Train: {processing_results['train_count']} files")
        print(f"  Val: {processing_results['val_count']} files")
        print(f"  Total points: {processing_results['total_points_generated']}")
        
        return processing_results

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

    def _split_files(self, mesh_paths):
        """
        Split mesh files into train and validation sets.
        
        Parameters:
            mesh_paths (list): List of mesh file paths
            
        Returns:
            tuple: (train_files, val_files)
        """
        import random
        
        # Handle edge cases
        if len(mesh_paths) == 0:
            return [], []
        elif len(mesh_paths) == 1:
            # If only one file, put it in training
            print("Warning: Only one file found, assigning to training set")
            return mesh_paths, []
        elif len(mesh_paths) == 2:
            # If only two files, one in each set
            print("Warning: Only two files found, one assigned to each set")
            return [mesh_paths[0]], [mesh_paths[1]]
        
        # For multiple files, use the split ratio
        # Sort paths for reproducible splits
        sorted_paths = sorted(mesh_paths)
        
        # Calculate split index
        train_count = max(1, int(len(sorted_paths) * self.train_val_split))
        
        # Ensure at least one file in validation if we have more than one file
        if train_count >= len(sorted_paths):
            train_count = len(sorted_paths) - 1
        
        train_files = sorted_paths[:train_count]
        val_files = sorted_paths[train_count:]
        
        print(f"Split: {len(train_files)} train, {len(val_files)} val (ratio: {self.train_val_split:.2f})")
        return train_files, val_files
    
    def _process_single_mesh(self, mesh_path, split='train'):
        """
        Process a single mesh file using the PointCloudProcessor logic.
        
        Parameters:
            mesh_path (str): Path to the mesh file
            split (str): Either 'train' or 'val' to determine output directory
            
        Returns:
            dict: Results for this specific mesh
        """
        try:
            # Load and normalize mesh using PointCloudProcessor method
            vertices, faces, name = self.point_cloud_processor.load_mesh(mesh_path)
        except ValueError as e:
            if "len(points)" in str(e) and "point_data" in str(e):
                print(f"  [{split.upper()}] [CORRUPTED] Skipping corrupted mesh file: {os.path.basename(mesh_path)}")
                print(f"    Error: {e}")
                return None
            else:
                # Re-raise if it's a different error
                raise
        except Exception as e:
            print(f"  [{split.upper()}] [ERROR] Failed to load mesh: {os.path.basename(mesh_path)}")
            print(f"    Error: {e}")
            return None
        
        # Determine output directory based on split
        if split == 'train':
            output_dir = self.train_data_path
        elif split == 'val':
            output_dir = self.val_data_path
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
        
        # Define output file paths
        points_file = os.path.join(output_dir, f"{name}_sampled_points.npy")
        distances_file = os.path.join(output_dir, f"{name}_signed_distances.npy")
        
        # Check if data already exists (same logic as PointCloudProcessor)
        if os.path.exists(points_file) and os.path.exists(distances_file):
            print(f"  [{split.upper()}] [Existing File] Loading data for {name}")
            sampled_points = np.load(points_file)
            distances = np.load(distances_file)
        else:
            print(f"  [{split.upper()}] [New File] Sampling points for {name}")
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
            'faces': faces,
            'split': split
        }
        
    def generate_sdf_dataset(self, z_dim=32, latent_mean=0.0, latent_sd=0.01):
        """
        Generate SDF dataset compatible with DeepSDF training pipeline.
        Uses both PointCloudProcessor and VolumeProcessor outputs.
        
        Parameters:
            z_dim (int): Latent vector dimension
            latent_mean (float): Mean for latent vector initialization
            latent_sd (float): Standard deviation for latent vector initialization
            
        Returns:
            dict: Dataset information for SDF training
        """
        # First ensure data is processed
        processing_results = self.process()
        
        # Get the device and resolution parameters from the VolumeProcessor
        device = self.volume_processor.device
        resolution = self.volume_processor.resolution
        
        # Get volume coordinates from VolumeProcessor
        volume_coords = self.volume_processor._get_volume_coords(device=device, resolution=resolution)
        
        # Create dataset metadata
        dataset_info = {
            'train_files': [],
            'val_files': [],
            'volume_coords': volume_coords,
            'dataset_params': {
                'z_dim': z_dim,
                'latent_mean': latent_mean,
                'latent_sd': latent_sd,
                'num_samples': self.n_uniform + self.n_gaussian * 10,
                'volume_coords_resolution': self.resolution,
                'point_cloud_params': processing_results['processing_params']
            },
            'processing_results': processing_results
        }
        
        # Collect train files
        for i, mesh_name in enumerate(processing_results['train_files']):
            points_file = processing_results['point_cloud_files']['train'][i]
            distances_file = processing_results['signed_distance_files']['train'][i]
            
            dataset_info['train_files'].append({
                'mesh_name': mesh_name,
                'points_file': points_file,
                'distances_file': distances_file,
                'split': 'train'
            })
        
        # Collect val files
        for i, mesh_name in enumerate(processing_results['val_files']):
            points_file = processing_results['point_cloud_files']['val'][i]
            distances_file = processing_results['signed_distance_files']['val'][i]
            
            dataset_info['val_files'].append({
                'mesh_name': mesh_name,
                'points_file': points_file,
                'distances_file': distances_file,
                'split': 'val'
            })
        
        print(f"Generated SDF dataset info:")
        print(f"  Train files: {len(dataset_info['train_files'])}")
        print(f"  Val files: {len(dataset_info['val_files'])}")
        print(f"  Total points: {processing_results['total_points_generated']}")
        print(f"  Volume resolution: {self.resolution}³")
        
        return dataset_info
    

    