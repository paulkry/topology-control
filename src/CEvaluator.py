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
        # Create test dataset (equivalent to ds_test from notebook)
        test_dataset = SDFDataset(dataset_info, split='val', fix_seed=True)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,  # Reproducible evaluation
            collate_fn=test_dataset.collate_fn
        )
        
        # Get volume coordinates (equivalent to all_coords, grad_size_axis from notebook)
        volume_processor = VolumeProcessor(device='cpu', resolution=self.resolution)
        all_coords = volume_processor._get_volume_coords(device='cpu', resolution=self.resolution)[0]
        grid_values = torch.arange(-1, 1, float(2/self.resolution))
        grad_size_axis = grid_values.shape[0]
        
        evaluation_results = {
            'evaluation_type': 'sdf_dataset',
            'test_losses': [],
            'extracted_meshes': [],
            'sdf_statistics': {'min_values': [], 'max_values': []},
            'resolution': self.resolution,
            'num_samples': len(test_dataset),
            'device': self.device
        }
        
        print(f"   Evaluating {len(test_dataset)} samples...")
        
        with torch.no_grad():
            # Evaluate losses on batched data
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (coords, latents, sdfs) in enumerate(test_loader):
                coords = coords.to(self.device)
                latents = latents.to(self.device)
                sdfs = sdfs.to(self.device)
                
                # Predict SDF values
                predicted_sdfs = self.model(latents, coords)
                predicted_sdfs = predicted_sdfs.squeeze(-1) if predicted_sdfs.dim() > 1 else predicted_sdfs
                
                # Calculate loss
                batch_loss = torch.nn.functional.mse_loss(predicted_sdfs, sdfs)
                evaluation_results['test_losses'].append(batch_loss.item())
                total_loss += batch_loss.item()
                num_batches += 1
            
            # Extract meshes for individual samples (equivalent to notebook extraction)
            print("   Extracting meshes from learned representations...")
            max_samples = min(3, len(test_dataset))  # Limit for faster evaluation
            
            for sample_idx in range(max_samples):
                coords, latent_vec, true_sdfs = test_dataset[sample_idx]
                
                # Get full volume prediction (equivalent to pred_full_sdfs from notebook)
                latent_vec = latent_vec.to(self.device)
                all_coords = all_coords.to(self.device)
                
                # Predict full SDF volume
                pred_full_sdfs = self.model(latent_vec.unsqueeze(0), all_coords.unsqueeze(0))
                pred_full_sdfs = pred_full_sdfs.squeeze()
                
                # Store statistics
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
                        'sdf_max': sdf_max,
                        'num_vertices': len(vertices),
                        'num_faces': len(faces)
                    })
                    
                    print(f"      Sample {sample_idx}: {len(vertices)} vertices, {len(faces)} faces")
                    
                except Exception as e:
                    print(f"      Sample {sample_idx}: Failed to extract mesh - {e}")
        
        # Compute final metrics
        evaluation_results['average_test_loss'] = total_loss / num_batches if num_batches > 0 else float('inf')
        
        if evaluation_results['sdf_statistics']['min_values']:
            evaluation_results['sdf_statistics']['avg_min'] = np.mean(evaluation_results['sdf_statistics']['min_values'])
            evaluation_results['sdf_statistics']['avg_max'] = np.mean(evaluation_results['sdf_statistics']['max_values'])
        
        evaluation_results['status'] = 'completed'
        
        print(f"   ✅ SDF evaluation complete!")
        print(f"      Average test loss: {evaluation_results['average_test_loss']:.6f}")
        print(f"      Meshes extracted: {len(evaluation_results['extracted_meshes'])}")
        
        return evaluation_results

    def extract_mesh(self, grad_size_axis, sdf, level=0.0):
        """
        Extract mesh from SDF using marching cubes.
        This is the same function from the notebook.
        """
        # Extract zero-level set with marching cubes
        grid_sdf = sdf.view(grad_size_axis, grad_size_axis, grad_size_axis).detach().cpu().numpy()
        vertices, faces, normals, _ = skimage.measure.marching_cubes(grid_sdf, level=level)

        # Rescale vertices extracted with marching cubes
        x_max = np.array([1, 1, 1])
        x_min = np.array([-1, -1, -1])
        vertices = vertices * ((x_max-x_min) / grad_size_axis) + x_min

        return vertices, faces

# Remove the duplicate evaluate method and the test code at the bottom