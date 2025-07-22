"""
Data Processing for Topology Control
"""
import os
import numpy as np
import meshio as meshio
import igl
import torch
from src.CGeometryUtils import PointCloudProcessor, VolumeProcessor, compute_volumes_from_mesh_files
  
class CDataProcessor:

    def __init__(self, config):
        """
        Initialize CDataProcessor with configuration.
        
        Args:
            config: Dictionary containing processor configuration
        """
        self.config = config
        
        # Extract paths from config
        self.raw_data_path = config['dataset_paths']['raw']
        self.processed_data_path = config['dataset_paths']['processed']
        self.dataset_type = config.get('dataset_type', 'sdf')
        
        # Create train/val subdirectories
        self.train_data_path = os.path.join(self.processed_data_path, 'train')
        self.val_data_path = os.path.join(self.processed_data_path, 'val')
        
        # Train/val split ratio
        self.train_val_split = config.get('train_val_split', 0.8)
        
        # NEW: Option to use same shapes for train and validation
        self.use_same_shapes_for_val = config.get('use_same_shapes_for_val', False)
        
        # Extract point cloud sampling parameters
        point_cloud_params = config.get('point_cloud_params', {})
        self.radius = point_cloud_params.get('radius', 0.02)
        self.sigma = point_cloud_params.get('sigma', 0.02)
        self.mu = point_cloud_params.get('mu', 0.0)
        self.n_gaussian = point_cloud_params.get('n_gaussian', 5)
        self.n_uniform = point_cloud_params.get('n_uniform', 10000)
        
        # Volume processing parameters
        volume_params = config.get('volume_processor_params', {})
        self.device = volume_params.get('device', 'cpu')
        self.resolution = volume_params.get('resolution', 50)
        
        # Initialize processors
        self.point_cloud_processor = PointCloudProcessor()
        self.volume_processor = VolumeProcessor(device=self.device, resolution=self.resolution)
        
        # Discover mesh files
        self.mesh_files = self._discover_mesh_files()
        
        # Calculate volumes if needed for VolumeSDFDataset
        if self.dataset_type == 'volumesdf':
            print("Computing volumes for VolumeSDFDataset...")
            self.shape_volumes, self.mesh_name_to_volume = compute_volumes_from_mesh_files(
                [os.path.join(self.raw_data_path, f) for f in self.mesh_files]
            )
            print(f"✓ Computed volumes for {len(self.shape_volumes)} meshes")
        else:
            self.shape_volumes = None
            self.mesh_name_to_volume = {}

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
            'corrupted_files': [],
            'skipped_files': [],
            'processing_errors': {},
            'point_cloud_files': {'train': [], 'val': []},
            'signed_distance_files': {'train': [], 'val': []},
            'total_points_generated': 0,
            'train_count': len(train_files),
            'val_count': len(val_files),
            'split_ratio': self.train_val_split,
            'use_same_shapes_for_val': self.use_same_shapes_for_val,  # Track this setting
            'processing_params': {
                'radius': self.radius,
                'sigma': self.sigma,
                'mu': self.mu,
                'n_gaussian': self.n_gaussian,
                'n_uniform': self.n_uniform
            },
            'shape_volumes': self.shape_volumes if hasattr(self, 'shape_volumes') else None
        }
        
        # Process training files
        print(f"Processing {len(train_files)} training files...")
        processed_train_meshes = set()  # Track processed meshes to avoid duplication
        
        for mesh_path in train_files:
            result = self._process_single_mesh(mesh_path, 'train')
            if result is not None:
                processing_results['processed_files'].append(result['mesh_name'])
                processing_results['train_files'].append(result['mesh_name'])
                processing_results['point_cloud_files']['train'].append(result['points_file'])
                processing_results['signed_distance_files']['train'].append(result['distances_file'])
                processing_results['total_points_generated'] += result['num_points']
                processed_train_meshes.add(result['mesh_name'])
            else:
                # Track failed files
                mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
                processing_results['corrupted_files'].append(mesh_name)
                processing_results['skipped_files'].append(f"{mesh_name} (train)")
        
        # Process validation files
        print(f"Processing {len(val_files)} validation files...")
        for mesh_path in val_files:
            mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
            
            # NEW: If using same shapes and mesh already processed for training, 
            # create symlinks or reference same files to save processing time
            if self.use_same_shapes_for_val and mesh_name in processed_train_meshes:
                print(f"  [VAL] [Same Shape] Reusing training data for {mesh_name}")
                
                # Find corresponding train files
                train_idx = processing_results['train_files'].index(mesh_name)
                train_points_file = processing_results['point_cloud_files']['train'][train_idx]
                train_distances_file = processing_results['signed_distance_files']['train'][train_idx]
                
                # For validation, we can either:
                # 1. Use same files (efficient but same exact samples)
                # 2. Process again with different random seed (different samples, more realistic validation)
                
                # Option 1: Reuse same files (faster)
                val_points_file = train_points_file
                val_distances_file = train_distances_file
                
                # Option 2: Process with different sampling (uncomment if preferred)
                # result = self._process_single_mesh(mesh_path, 'val')
                # if result is not None:
                #     val_points_file = result['points_file']
                #     val_distances_file = result['distances_file']
                # else:
                #     continue
                
                processing_results['val_files'].append(mesh_name)
                processing_results['point_cloud_files']['val'].append(val_points_file)
                processing_results['signed_distance_files']['val'].append(val_distances_file)
                
                # Don't double-count points if reusing same files
                if val_points_file != train_points_file:
                    points_data = np.load(val_points_file)
                    processing_results['total_points_generated'] += len(points_data)
            
            else:
                # Process normally for different shapes or if not reusing
                result = self._process_single_mesh(mesh_path, 'val')
                if result is not None:
                    if mesh_name not in processing_results['processed_files']:
                        processing_results['processed_files'].append(result['mesh_name'])
                    processing_results['val_files'].append(result['mesh_name'])
                    processing_results['point_cloud_files']['val'].append(result['points_file'])
                    processing_results['signed_distance_files']['val'].append(result['distances_file'])
                    processing_results['total_points_generated'] += result['num_points']
                else:
                    # Track failed files
                    processing_results['corrupted_files'].append(mesh_name)
                    processing_results['skipped_files'].append(f"{mesh_name} (val)")
        
        # Update counts to reflect actual processed files
        processing_results['train_count'] = len(processing_results['train_files'])
        processing_results['val_count'] = len(processing_results['val_files'])
        processing_results['success_rate'] = len(set(processing_results['processed_files'])) / len(mesh_paths)
        
        # Summary logging
        print(f"Data processing complete.")
        print(f"  Successfully processed: {len(set(processing_results['processed_files']))} unique files")
        print(f"  Train: {processing_results['train_count']} files")
        print(f"  Val: {processing_results['val_count']} files")
        
        if self.use_same_shapes_for_val:
            print(f"  📋 Using same shapes for train and validation (DeepSDF overfitting mode)")
        
        print(f"  Total points: {processing_results['total_points_generated']}")
        
        if processing_results['corrupted_files']:
            print(f"  ⚠️  Corrupted/skipped: {len(processing_results['corrupted_files'])} files")
            print(f"  Success rate: {processing_results['success_rate']:.1%}")
        
        return processing_results

    def generate_sdf_dataset(self, z_dim=32, latent_mean=0.0, latent_sd=0.01):
        """
        Generate SDF dataset compatible with DeepSDF training pipeline.
        For VolumeSDFDataset, includes shape volume information.
        """
        # First ensure data is processed
        processing_results = self.process()
        
        # Get the device and resolution parameters from the VolumeProcessor
        device = self.volume_processor.device
        resolution = self.volume_processor.resolution
        
        # Get volume coordinates from VolumeProcessor
        volume_coords = self.volume_processor.volume_coords
        
        # Convert volume_coords tensor to numpy array (JSON serializable)
        if hasattr(volume_coords, 'cpu'):
            volume_coords_array = volume_coords.cpu().numpy()
        else:
            volume_coords_array = volume_coords
        
        # Create dataset metadata - only use serializable formats
        dataset_info = {
            'train_files': [],
            'val_files': [],
            'volume_coords': volume_coords_array,
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
        
        # Prepare volume data for VolumeSDFDataset
        if self.shape_volumes is not None:
            train_volumes = []
            val_volumes = []
            
            # Collect train files with volumes
            for i, mesh_name in enumerate(processing_results['train_files']):
                points_file = processing_results['point_cloud_files']['train'][i]
                distances_file = processing_results['signed_distance_files']['train'][i]
                
                # Get volume for this mesh
                volume = self.mesh_name_to_volume.get(mesh_name, 0.0)
                train_volumes.append(volume)
                
                dataset_info['train_files'].append({
                    'mesh_name': mesh_name,
                    'points_file': points_file,
                    'distances_file': distances_file,
                    'volume': volume,
                    'split': 'train'
                })
            
            # Collect val files with volumes
            for i, mesh_name in enumerate(processing_results['val_files']):
                points_file = processing_results['point_cloud_files']['val'][i]
                distances_file = processing_results['signed_distance_files']['val'][i]
                
                # Get volume for this mesh
                volume = self.mesh_name_to_volume.get(mesh_name, 0.0)
                val_volumes.append(volume)
                
                dataset_info['val_files'].append({
                    'mesh_name': mesh_name,
                    'points_file': points_file,
                    'distances_file': distances_file,
                    'volume': volume,
                    'split': 'val'
                })
            
            # FIXED: Add split-specific volume arrays for VolumeSDFDataset
            dataset_info['shape_volumes'] = self.shape_volumes  # All volumes (for reference)
            dataset_info['train_volumes'] = train_volumes       # Train split volumes - KEY FIX
            dataset_info['val_volumes'] = val_volumes           # Val split volumes - KEY FIX
            
            # Volume statistics for normalization/analysis
            all_volumes = np.array(self.shape_volumes)
            dataset_info['volume_stats'] = {
                'min_volume': float(np.min(all_volumes)),
                'max_volume': float(np.max(all_volumes)),
                'mean_volume': float(np.mean(all_volumes)),
                'std_volume': float(np.std(all_volumes)),
                'total_samples': len(all_volumes),
                'train_samples': len(train_volumes),
                'val_samples': len(val_volumes)
            }
            
            print(f"Generated VolumeSDFDataset info:")
            print(f"  Train files: {len(dataset_info['train_files'])} (volumes: {len(train_volumes)})")
            print(f"  Val files: {len(dataset_info['val_files'])} (volumes: {len(val_volumes)})")
            print(f"  Volume range: [{dataset_info['volume_stats']['min_volume']:.6f}, {dataset_info['volume_stats']['max_volume']:.6f}]")
            
        else:
            # Regular SDF dataset without volumes
            for i, mesh_name in enumerate(processing_results['train_files']):
                points_file = processing_results['point_cloud_files']['train'][i]
                distances_file = processing_results['signed_distance_files']['train'][i]
                
                dataset_info['train_files'].append({
                    'mesh_name': mesh_name,
                    'points_file': points_file,
                    'distances_file': distances_file,
                    'split': 'train'
                })
            
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
        print(f"  Volume coords shape: {volume_coords_array.shape}")
        
        # Return in the format expected by the pipeline orchestrator
        return {
            'status': 'success',
            'dataset_info': dataset_info,
            'processed_data_path': self.processed_data_path,
            'train_files': dataset_info['train_files'],
            'val_files': dataset_info['val_files'],
            'processing_results': processing_results,
            'total_points_generated': processing_results['total_points_generated']
        }
    
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
            # If only one file, put it in training (and validation if same_shapes is True)
            if self.use_same_shapes_for_val:
                print("Using single file for both training and validation (DeepSDF style)")
                return mesh_paths, mesh_paths
            else:
                print("Warning: Only one file found, assigning to training set")
                return mesh_paths, []
        
        # NEW: Handle same shapes for train and validation
        if self.use_same_shapes_for_val:
            print(f"Using same {len(mesh_paths)} shapes for both training and validation (DeepSDF overfitting)")
            # Sort paths for reproducible ordering
            sorted_paths = sorted(mesh_paths)
            return sorted_paths, sorted_paths
        
        # Original split logic for different train/val sets
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
