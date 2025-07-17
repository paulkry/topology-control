"""
Evaluate the SDF model and extract meshes from learned representations.
"""
import torch
import numpy as np
import skimage.measure
from torch.utils.data import DataLoader
from CModelTrainer import SDFDataset
from CGeometryUtils import VolumeProcessor

class CEvaluator:
    def __init__(self, model, device='cpu'):
        """
        Initialize the evaluator with a trained SDF model.
        
        Parameters:
            model: Trained SDF model (e.g., DeepSDF)
            device: Device to run evaluation on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def evaluate_sdf_dataset(self, dataset_info, batch_size=16, resolution=50):
        """
        Evaluate the SDF model on a test dataset and extract meshes.
        
        Parameters:
            dataset_info: Dataset info from CDataProcessor.generate_sdf_dataset()
            batch_size: Batch size for evaluation
            resolution: Resolution for volume grid evaluation
            
        Returns:
            dict: Evaluation results including losses and extracted meshes
        """
        # Create test dataset (equivalent to ds_test from notebook)
        test_dataset = SDFDataset(dataset_info, split='val', fix_seed=True)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,  # Changed from True for reproducible evaluation
            collate_fn=test_dataset.collate_fn
        )
        
        # Get volume coordinates (equivalent to all_coords, grad_size_axis from notebook)
        volume_processor = VolumeProcessor(device='cpu', resolution=resolution)
        device = volume_processor.device
        resolution = volume_processor.resolution
        all_coords = volume_processor._get_volume_coords(device=device, resolution=resolution)[0]
        grid_values = torch.arange(-1, 1, float(2/resolution))
        grad_size_axis = grid_values.shape[0]
        
        evaluation_results = {
            'test_losses': [],
            'extracted_meshes': [],
            'sdf_statistics': {'min_values': [], 'max_values': []},
            'resolution': resolution,
            'num_samples': len(test_dataset)
        }
        
        print(f"Evaluating {len(test_dataset)} samples...")
        
        with torch.no_grad():
            # Evaluate losses on batched data
            for batch_idx, (coords, latents, sdfs) in enumerate(test_loader):
                coords = coords.to(self.device)
                latents = latents.to(self.device)
                sdfs = sdfs.to(self.device)
                
                # Fix: Concatenate latents and coords for DeepSDF input (same as in CModelTrainer)
                if coords.dim() == 3:
                    batch_size, num_points, coord_dim = coords.shape
                    coords = coords.view(-1, coord_dim)
                    latents = latents.unsqueeze(1).expand(-1, num_points, -1).contiguous()
                    latents = latents.view(-1, latents.shape[-1])
                    sdfs = sdfs.view(-1)
                
                model_input = torch.cat([latents, coords], dim=1)
                predicted_sdfs = self.model(model_input)
                predicted_sdfs = predicted_sdfs.squeeze(-1) if predicted_sdfs.dim() > 1 else predicted_sdfs
                
                batch_loss = torch.nn.functional.mse_loss(predicted_sdfs, sdfs)
                evaluation_results['test_losses'].append(batch_loss.item())
            
            # Extract meshes for individual samples (equivalent to notebook extraction)
            print("Extracting meshes from learned representations...")
            for sample_idx in range(min(3, len(test_dataset))):  # Reduced to 3 for faster testing
                coords, latent_vec, true_sdfs = test_dataset[sample_idx]
                
                # Get full volume prediction (equivalent to pred_full_sdfs from notebook)
                latent_vec = latent_vec.to(self.device)
                all_coords = all_coords.to(self.device)
                
                # Expand latent vector to match all coordinates
                num_coords = all_coords.shape[0]
                latent_expanded = latent_vec.unsqueeze(0).expand(num_coords, -1)
                
                # Concatenate for model input
                model_input = torch.cat([latent_expanded, all_coords], dim=1)
                pred_full_sdfs = self.model(model_input)
                pred_full_sdfs = pred_full_sdfs.squeeze(-1)
                
                # Store statistics (equivalent to pred_full_sdfs.max(), pred_full_sdfs.min())
                sdf_min = pred_full_sdfs.min().item()
                sdf_max = pred_full_sdfs.max().item()
                evaluation_results['sdf_statistics']['min_values'].append(sdf_min)
                evaluation_results['sdf_statistics']['max_values'].append(sdf_max)
                
                # Extract mesh (equivalent to vertices_pred, faces_pred from notebook)
                try:
                    vertices, faces = self.extract_mesh(grad_size_axis, pred_full_sdfs)
                    
                    evaluation_results['extracted_meshes'].append({
                        'sample_idx': sample_idx,
                        'vertices': vertices,
                        'faces': faces,
                        'sdf_min': sdf_min,
                        'sdf_max': sdf_max
                    })
                    
                    print(f"    Sample {sample_idx}: {len(vertices)} vertices, {len(faces)} faces")
                    
                except Exception as e:
                    print(f"    Sample {sample_idx}: Failed to extract mesh - {e}")
        
        # Compute average metrics
        evaluation_results['average_test_loss'] = np.mean(evaluation_results['test_losses'])
        evaluation_results['sdf_statistics']['avg_min'] = np.mean(evaluation_results['sdf_statistics']['min_values'])
        evaluation_results['sdf_statistics']['avg_max'] = np.mean(evaluation_results['sdf_statistics']['max_values'])
        
        return evaluation_results

    def extract_mesh(self, grad_size_axis, sdf, level=0.0):
        """
        Extract mesh from SDF using marching cubes.
        This is the same function from the notebook, now properly integrated.
        
        Parameters:
            grad_size_axis: Grid size along each axis
            sdf: SDF tensor 
            level: Iso-surface level (default 0.0)
            
        Returns:
            tuple: (vertices, faces)
        """
        # Extract zero-level set with marching cubes
        grid_sdf = sdf.view(grad_size_axis, grad_size_axis, grad_size_axis).detach().cpu().numpy()
        vertices, faces, normals, _ = skimage.measure.marching_cubes(grid_sdf, level=level)

        # Rescale vertices extracted with marching cubes
        x_max = np.array([1, 1, 1])
        x_min = np.array([-1, -1, -1])
        vertices = vertices * ((x_max-x_min) / grad_size_axis) + x_min

        return vertices, faces

    def evaluate(self, dataset_info, **kwargs):
        """
        Main evaluation method for SDF datasets.
        
        Parameters:
            dataset_info: Dataset info for SDF evaluation (required)
            **kwargs: Additional parameters for evaluation
            
        Returns:
            Evaluation results
        """
        return self.evaluate_sdf_dataset(dataset_info, **kwargs)
    
# Test the CEvaluator class
if __name__ == "__main__":
    print("Testing CEvaluator with real data...")
    
    # First, generate processed data and train a model
    from CDataProcessor import CDataProcessor
    from CModelTrainer import CModelTrainer
    from CArchitectureManager import DeepSDF
    
    # Configuration for data processing
    data_config = {
        'dataset_paths': {
            'raw': 'data/raw',
            'processed': 'data/processed'
        },
        'train_val_split': 0.7,  # 70% train, 30% validation
        'point_cloud_params': {
            'radius': 0.02,
            'sigma': 0.01,
            'mu': 0.0,
            'n_gaussian': 3,  # Fewer samples for faster testing
            'n_uniform': 500  # Fewer samples for faster testing
        },
        'volume_processor_params': {
            'device': 'cpu',  # Use CPU for compatibility
            'resolution': 16   # Smaller resolution for faster testing
        }
    }
    
    print("Step 1: Processing data with CDataProcessor...")
    try:
        processor = CDataProcessor(data_config)
        print(f"Found {len(processor.mesh_files)} mesh files: {processor.mesh_files}")
        
        # Generate SDF dataset
        dataset_info = processor.generate_sdf_dataset(
            z_dim=32,  # Smaller latent dimension for testing
            latent_mean=0.0,
            latent_sd=0.01
        )
        print(f"✓ Data processing complete!")
        print(f"  Train files: {len(dataset_info['train_files'])}")
        print(f"  Val files: {len(dataset_info['val_files'])}")
        
    except Exception as e:
        print(f" Error in data processing: {e}")
        print("Make sure you have mesh files in data/raw/")
        exit(1)
    
    print("\nStep 2: Quick model training...")
    
    # Configuration for model training (minimal for testing)
    trainer_config = {
        'processed_data_path': 'data/processed',
        'dataset_type': 'sdf',
        'num_epochs': 2,  # Very few epochs for quick testing
        'batch_size': 2,   # Small batch size
        'learning_rate': 0.001,
        'max_points': 5000,
        'optimizer': 'adam',
        'loss_function': 'mse',
        'sdf_delta': 1.0,
        'latent_sd': 0.01
    }
    
    trainer = CModelTrainer(trainer_config)
    
    # Create a config for the model
    model_config = {
        'z_dim': 32,  # Match the dataset latent dimension
        'layer_size': 64,  # Small network for testing
        'coord_dim': 3,  # 3D coordinates
        'dropout_p': 0.1
    }
    
    try:
        model = DeepSDF(model_config)
        print("✓ Model created successfully")
        
        print("  Training model (this may take a moment)...")
        trained_model, report = trainer.train_and_validate(model, dataset_info)
        
        if "error" in report:
            print(f" Training failed: {report['error']}")
            exit(1)
        else:
            print(f"✓ Training completed! Final val loss: {report['final_val_loss']:.6f}")
        
    except Exception as e:
        print(f" Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\nStep 3: Testing CEvaluator...")
    
    try:
        # Create evaluator with the trained model
        evaluator = CEvaluator(trained_model, device='cpu')
        print("✓ Evaluator created successfully")
        
        print("  Running evaluation on validation set...")
        evaluation_results = evaluator.evaluate_sdf_dataset(
            dataset_info,
            batch_size=2,      # Small batch size for testing
            resolution=16      # Low resolution for faster mesh extraction
        )
        
        print("\n" + "="*50)
        print("Evaluation Completed!")
        print("="*50)
        print(f"✓ Evaluation successful!")
        print(f"  Samples evaluated: {evaluation_results['num_samples']}")
        print(f"  Average test loss: {evaluation_results['average_test_loss']:.6f}")
        print(f"  Meshes extracted: {len(evaluation_results['extracted_meshes'])}")
        
        # Print SDF statistics
        sdf_stats = evaluation_results['sdf_statistics']
        print(f"  SDF value range: [{sdf_stats['avg_min']:.3f}, {sdf_stats['avg_max']:.3f}]")
        
        # Print mesh details
        if evaluation_results['extracted_meshes']:
            print("  Extracted mesh details:")
            for mesh_info in evaluation_results['extracted_meshes']:
                print(f"    Sample {mesh_info['sample_idx']}: "
                      f"{len(mesh_info['vertices'])} vertices, "
                      f"{len(mesh_info['faces'])} faces")
        
        print(f"  Evaluation resolution: {evaluation_results['resolution']}³ grid")
        
    except Exception as e:
        print(f" Error during evaluation: {e}")
        import traceback
        traceback.print_exc()