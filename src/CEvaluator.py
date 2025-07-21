"""
Evaluate the SDF model and extract meshes from learned representations.
"""
import torch
import numpy as np
import skimage.measure
from torch.utils.data import DataLoader
from src.CModelTrainer import SDFDataset
from src.CGeometryUtils import VolumeProcessor

class CEvaluator:
    def __init__(self, eval_config, processor_config):
        """
        Initialize the evaluator with a trained SDF model.
        
        Parameters:
            model: Trained SDF model (e.g., DeepSDF)
            device: Device to run evaluation on ('cpu' or 'cuda')
        """
        self.processor_config = processor_config
        self.config = eval_config

    def evaluate(self, model, dataset_info, batch_size=16, resolution=50):
        """
        Evaluate the SDF model on a test dataset and extract meshes.
        
        Parameters:
            dataset_info: Dataset info from CDataProcessor.generate_sdf_dataset()
            
        Returns:
            dict: Evaluation results including losses and extracted meshes
        """
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        self.resolution = resolution

        print(f"Using device: {self.device}")
        print(f"Evaluation resolution: {resolution}")
        
        # Determine dataset type based on model architecture
        model_name = model.__class__.__name__
        is_volume_model = model_name == 'VolumeDeepSDF'
        
        # Create test dataset - use VolumeSDFDataset for VolumeDeepSDF
        if is_volume_model:
            from src.CModelTrainer import VolumeSDFDataset
            test_dataset = VolumeSDFDataset(dataset_info, split='val', fix_seed=True)
            print(f"   Using VolumeSDFDataset for {model_name}")
        else:
            test_dataset = SDFDataset(dataset_info, split='val', fix_seed=True)
            print(f"   Using SDFDataset for {model_name}")
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,  # Reproducible evaluation
            collate_fn=test_dataset.collate_fn
        )
        
        # Get volume coordinates - FIX: Use consistent resolution calculation
        volume_processor = VolumeProcessor(device='cpu', resolution=self.resolution)
        all_coords, actual_grid_size = volume_processor._get_volume_coords(device='cpu', resolution=self.resolution)
        
        print(f"   Volume coordinates: {all_coords.shape}")
        print(f"   Actual grid size: {actual_grid_size}")
        print(f"   Expected volume size: {actual_grid_size ** 3}")
        
        evaluation_results = {
            'evaluation_type': 'volumesdf_dataset' if is_volume_model else 'sdf_dataset',
            'model_type': model_name,
            'test_losses': [],
            'volume_losses': [] if is_volume_model else None,
            'sdf_losses': [],
            'extracted_meshes': [],
            'sdf_statistics': {'min_values': [], 'max_values': []},
            'volume_statistics': {'predictions': [], 'targets': []} if is_volume_model else None,
            'resolution': self.resolution,
            'actual_grid_size': actual_grid_size,
            'num_samples': len(test_dataset),
            'device': self.device
        }
        
        print(f"   Evaluating {len(test_dataset)} samples...")
        
        with torch.no_grad():
            # Evaluate losses on batched data
            total_loss = 0.0
            total_sdf_loss = 0.0
            total_volume_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch_data in enumerate(test_loader):
                if is_volume_model:
                    # VolumeSDFDataset returns 4 items: coords, latents, sdfs, volumes
                    coords, latents, sdfs, volumes = batch_data
                    coords = coords.to(self.device)
                    latents = latents.to(self.device)
                    sdfs = sdfs.to(self.device)
                    volumes = volumes.to(self.device)
                    
                    # Predict both SDF and volume
                    predictions = self.model(latents, coords)
                    predicted_sdfs = predictions['sdf']
                    predicted_volumes = predictions['volume']
                    
                    # Calculate losses
                    sdf_loss = torch.nn.functional.mse_loss(predicted_sdfs, sdfs)
                    volume_loss = torch.nn.functional.mse_loss(predicted_volumes, volumes)
                    combined_loss = sdf_loss + volume_loss  # Simple combination for evaluation
                    
                    evaluation_results['sdf_losses'].append(sdf_loss.item())
                    evaluation_results['volume_losses'].append(volume_loss.item())
                    evaluation_results['test_losses'].append(combined_loss.item())
                    
                    # Store volume statistics
                    evaluation_results['volume_statistics']['predictions'].extend(predicted_volumes.cpu().numpy().tolist())
                    evaluation_results['volume_statistics']['targets'].extend(volumes.cpu().numpy().tolist())
                    
                    total_sdf_loss += sdf_loss.item()
                    total_volume_loss += volume_loss.item()
                    total_loss += combined_loss.item()
                    
                else:
                    # Regular SDFDataset returns 3 items: coords, latents, sdfs
                    coords, latents, sdfs = batch_data
                    coords = coords.to(self.device)
                    latents = latents.to(self.device)
                    sdfs = sdfs.to(self.device)
                    
                    # Predict SDF values
                    predicted_sdfs = self.model(latents, coords)
                    predicted_sdfs = predicted_sdfs.squeeze(-1) if predicted_sdfs.dim() > 1 else predicted_sdfs
                    
                    # Calculate loss
                    batch_loss = torch.nn.functional.mse_loss(predicted_sdfs, sdfs)
                    evaluation_results['test_losses'].append(batch_loss.item())
                    evaluation_results['sdf_losses'].append(batch_loss.item())
                    
                    total_loss += batch_loss.item()
                    total_sdf_loss += batch_loss.item()
                
                num_batches += 1
            
            # Extract meshes for individual samples (equivalent to notebook extraction)
            print("   Extracting meshes from learned representations...")
            max_samples = min(3, len(test_dataset))  # Limit for faster evaluation
            
            for sample_idx in range(max_samples):
                sample_data = test_dataset[sample_idx]
                
                if is_volume_model:
                    coords, latent_vec, true_sdfs, true_volume = sample_data
                    print(f"      Sample {sample_idx} (Volume Model):")
                    print(f"        True volume: {true_volume:.6f}")
                else:
                    coords, latent_vec, true_sdfs = sample_data
                    print(f"      Sample {sample_idx} (SDF Model):")
                
                print(f"        Latent shape: {latent_vec.shape}")
                print(f"        Sample coords shape: {coords.shape}")
                
                # Get full volume prediction (equivalent to pred_full_sdfs from notebook)
                latent_vec = latent_vec.to(self.device)
                all_coords = all_coords.to(self.device)
                
                # FIX: Proper model input formatting
                latent_batch = latent_vec.unsqueeze(0)  # [1, latent_dim]
                coords_batch = all_coords.unsqueeze(0)  # [1, num_coords, 3]
                
                print(f"        Model inputs - latent: {latent_batch.shape}, coords: {coords_batch.shape}")
                
                # Predict full SDF volume
                if is_volume_model:
                    predictions = self.model(latent_batch, coords_batch)
                    pred_full_sdfs = predictions['sdf'].squeeze()
                    pred_volume = predictions['volume'].squeeze()
                    print(f"        Predicted volume: {pred_volume.item():.6f}")
                else:
                    pred_full_sdfs = self.model(latent_batch, coords_batch)
                    pred_full_sdfs = pred_full_sdfs.squeeze()  # Remove batch dimension
                
                print(f"        Model output shape: {pred_full_sdfs.shape}")
                print(f"        Expected size: {actual_grid_size ** 3}")
                
                # Validate tensor size before proceeding
                if pred_full_sdfs.numel() != actual_grid_size ** 3:
                    print(f"        ❌ Size mismatch: got {pred_full_sdfs.numel()}, expected {actual_grid_size ** 3}")
                    print(f"        Skipping sample {sample_idx}")
                    continue
                
                # Store statistics
                sdf_min = pred_full_sdfs.min().item()
                sdf_max = pred_full_sdfs.max().item()
                evaluation_results['sdf_statistics']['min_values'].append(sdf_min)
                evaluation_results['sdf_statistics']['max_values'].append(sdf_max)
                
                print(f"        SDF range: [{sdf_min:.4f}, {sdf_max:.4f}]")
                
                # Extract mesh (equivalent to vertices_pred, faces_pred from notebook)
                try:
                    # FIX: Pass the SDF tensor directly, not with unsqueeze
                    vertices, faces = self.extract_mesh(actual_grid_size, pred_full_sdfs)
                    
                    mesh_info = {
                        'sample_idx': sample_idx,
                        'vertices': vertices,
                        'faces': faces,
                        'sdf_min': sdf_min,
                        'sdf_max': sdf_max,
                        'num_vertices': len(vertices),
                        'num_faces': len(faces)
                    }
                    
                    # Add volume info for VolumeDeepSDF
                    if is_volume_model:
                        mesh_info['predicted_volume'] = pred_volume.item()
                        mesh_info['true_volume'] = true_volume
                    
                    evaluation_results['extracted_meshes'].append(mesh_info)
                    
                    print(f"        ✅ Mesh extracted: {len(vertices)} vertices, {len(faces)} faces")
                    
                    # Visualize the mesh
                    self.visualize_mesh(vertices, faces, coords.cpu().numpy(), true_sdfs.cpu().numpy(), f"Sample_{sample_idx}")
                    
                except Exception as e:
                    print(f"        ❌ Failed to extract mesh: {e}")
                    import traceback
                    print(f"        Traceback: {traceback.format_exc()}")
        
        # Compute final metrics
        evaluation_results['average_test_loss'] = total_loss / num_batches if num_batches > 0 else float('inf')
        evaluation_results['average_sdf_loss'] = total_sdf_loss / num_batches if num_batches > 0 else float('inf')
        
        if is_volume_model:
            evaluation_results['average_volume_loss'] = total_volume_loss / num_batches if num_batches > 0 else float('inf')
            
            # Volume prediction statistics
            if evaluation_results['volume_statistics']['predictions']:
                pred_vols = np.array(evaluation_results['volume_statistics']['predictions'])
                true_vols = np.array(evaluation_results['volume_statistics']['targets'])
                evaluation_results['volume_statistics']['mae'] = np.mean(np.abs(pred_vols - true_vols))
                evaluation_results['volume_statistics']['rmse'] = np.sqrt(np.mean((pred_vols - true_vols) ** 2))
        
        if evaluation_results['sdf_statistics']['min_values']:
            evaluation_results['sdf_statistics']['avg_min'] = np.mean(evaluation_results['sdf_statistics']['min_values'])
            evaluation_results['sdf_statistics']['avg_max'] = np.mean(evaluation_results['sdf_statistics']['max_values'])
        
        evaluation_results['status'] = 'completed'
        
        print(f"   ✅ SDF evaluation complete!")
        print(f"      Average SDF loss: {evaluation_results['average_sdf_loss']:.6f}")
        if is_volume_model:
            print(f"      Average volume loss: {evaluation_results['average_volume_loss']:.6f}")
            print(f"      Average combined loss: {evaluation_results['average_test_loss']:.6f}")
            if 'mae' in evaluation_results['volume_statistics']:
                print(f"      Volume MAE: {evaluation_results['volume_statistics']['mae']:.6f}")
                print(f"      Volume RMSE: {evaluation_results['volume_statistics']['rmse']:.6f}")
        print(f"      Meshes extracted: {len(evaluation_results['extracted_meshes'])}")
        
        return evaluation_results

    def extract_mesh(self, grad_size_axis, sdf, level=0.0):
        """
        Extract mesh from SDF using marching cubes.
        """
        print(f"        extract_mesh: grid_size={grad_size_axis}, sdf_shape={sdf.shape}, sdf_numel={sdf.numel()}")
        
        try:
            # Validate input size
            expected_size = grad_size_axis ** 3
            if sdf.numel() != expected_size:
                raise ValueError(f"SDF size mismatch: got {sdf.numel()}, expected {expected_size} for {grad_size_axis}³ grid")
            
            # Check minimum grid size for marching cubes
            if grad_size_axis < 2:
                raise ValueError(f"Grid size {grad_size_axis} is too small. Marching cubes requires at least 2x2x2 grid.")
            
            # Extract zero-level set with marching cubes
            grid_sdf = sdf.view(grad_size_axis, grad_size_axis, grad_size_axis).detach().cpu().numpy()
            print(f"        Grid SDF shape after reshape: {grid_sdf.shape}")
            print(f"        Grid SDF range: [{grid_sdf.min():.4f}, {grid_sdf.max():.4f}]")
            
            # Automatically adjust level if it's outside the SDF range
            sdf_min, sdf_max = grid_sdf.min(), grid_sdf.max()
            original_level = level
            
            if level < sdf_min or level > sdf_max:
                if sdf_min < 0 < sdf_max:
                    level = 0.0
                    print(f"        Using zero level set (level=0.0)")
                else:
                    # Use a level that's within the range, typically the median or a small value
                    level = np.percentile(grid_sdf, 20)  # 20th percentile often works well
                    print(f"        Level {original_level} outside range [{sdf_min:.4f}, {sdf_max:.4f}]")
                    print(f"        Using level={level:.4f} (20th percentile)")
            
            vertices, faces, normals, _ = skimage.measure.marching_cubes(grid_sdf, level=level)
            print(f"        Marching cubes extracted: {len(vertices)} vertices, {len(faces)} faces")

            # Rescale vertices extracted with marching cubes
            x_max = np.array([1, 1, 1])
            x_min = np.array([-1, -1, -1])
            vertices = vertices * ((x_max-x_min) / grad_size_axis) + x_min

            return vertices, faces
            
        except Exception as e:
            print(f"        extract_mesh error: {e}")
            return np.array([]), np.array([])
        
    def visualize_mesh(self, vertices, faces, sampled_points=None, distances=None, name="mesh"):
        """
        Visualize the extracted mesh using polyscope.
        """
        try:
            # Skip visualization if mesh is empty
            if len(vertices) == 0 or len(faces) == 0:
                print(f"        Skipping visualization of empty mesh: {name}")
                return
            
            if not hasattr(self, 'renderer'):
                import polyscope as ps
                ps.init()
                self.renderer = ps
            
            # Register the surface mesh
            mesh = self.renderer.register_surface_mesh(name, vertices, faces)
            
            # Add point cloud if provided
            if sampled_points is not None:
                point_cloud_name = f"{name}_points"
                point_cloud = self.renderer.register_point_cloud(point_cloud_name, sampled_points, radius=0.01)
                
                # Add scalar quantities to the point cloud (not the main polyscope module)
                if distances is not None:
                    point_cloud.add_scalar_quantity("distances", distances)
            
            print(f"        Visualization registered: {name}")
            self.renderer.show()
            
        except ImportError:
            print(f"        Polyscope not available. Install with: pip install polyscope")
        except Exception as e:
            print(f"        Visualization failed for {name}: {e}")