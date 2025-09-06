from deepsdf.Trainer import ModelTrainer
from deepsdf.ArtifactManager import ArtifactManager
from deepsdf.Pointcloud import PointCloudProcessor
from deepsdf.Model import DeepSDF

import yaml
import os
import traceback
import numpy as np

class PipelineOrchestrator:
    def __init__(self, config_file="config/config_examples.yaml"):
        self.config = self._load_config(config_file)
        self._current_step = None

        # Initialize artifact manager first to get experiment info
        self.artifact_manager = ArtifactManager(self.config["artifacts_config"])

        # Get experiment info for component integration
        experiment_summary = self.artifact_manager.get_experiment_summary()
        experiment_id = experiment_summary['experiment_id']
        artifacts_base = self.config["artifacts_config"]["save_artifacts_to"]

        self.pointcloud_processor = None  # Will initialize lazily once paths known

        # Pass experiment info to trainer for integrated artifact saving
        trainer_config = self.config["trainer_config"].copy()
        trainer_config['experiment_id'] = experiment_id
        trainer_config['artifacts_base'] = artifacts_base
        self.model_trainer = ModelTrainer(trainer_config)
        
    def run(self):
        """Execute the ML pipeline: Data → Model → Training → Saving"""
        pipeline_state = {}
        
        try:
            self._current_step = 'data_processing'
            self._process_data_step(pipeline_state)
            
            self._current_step = 'model_building'
            self._build_model_step(pipeline_state)
            
            self._current_step = 'training'
            self._train_model_step(pipeline_state)
                        
            self._save_pipeline_summary(pipeline_state)
            print("\n=== Pipeline Completed Successfully ===\n")
            
        except Exception as e:
            self._handle_pipeline_error(e)
            raise

    def _load_config(self, config_file):
        with open(config_file, "r") as file:
            print("\n=== Configuration Loaded ===\n")
            config = yaml.safe_load(file)

        home = config.get('home')
        if not home:
            raise ValueError("The 'home' path is missing from the configuration.")

        self._resolve_config_paths(config, home)
        return config

    def _resolve_config_paths(self, config, home):
        """Convert relative paths to absolute paths using the home directory."""
        def join_with_home(path):
            if os.path.isabs(path):
                return path
            return os.path.join(home, path)

        # Define path mappings for each config section
        path_mappings = {
            'artifacts_config': ['save_artifacts_to'],
            'trainer_config': ['fine_tune_model_path', 'fine_tune_data_path']
        }

        # Update paths for each config section
        for section_name, path_keys in path_mappings.items():
            section = config.get(section_name, {})
            for path_key in path_keys:
                if path_key in section:
                    section[path_key] = join_with_home(section[path_key])

        # Handle dataset_paths separately as it's nested
        processor = config.get('processor_config', {})
        dataset_paths = processor.get('dataset_paths', {})
        for key, value in dataset_paths.items():
            dataset_paths[key] = join_with_home(value)

    def _handle_pipeline_error(self, error):
        """Handle pipeline errors and save error information."""
        print(f"\nPipeline FAILED at step '{self._current_step}': {error}")
        error_info = {
            'error': str(error),
            'failed_step': self._current_step,
            'traceback': traceback.format_exc()
        }
        self.artifact_manager.save_artifacts(error_report=error_info)

    def _should_skip_step(self, config_section, skip_key, default=True):
        """Check if a pipeline step should be skipped based on configuration."""
        return self.config.get(config_section, {}).get(skip_key, default)

    def _process_data_step(self, state):
        """Step 1: Process and prepare data for training."""
        if self._should_skip_step('processor_config', 'skip_processing'):
            print("\nStep 1: Data Processing - SKIPPED")
            return
        print("\nStep 1: Processing Data (PointCloud + SDF Generation)...")

        trainer_config = self.config.get('trainer_config', {})
        processor_config = self.config.get('processor_config', {})
        # Migrate legacy top-level point_cloud_params if present
        if 'point_cloud_params' in self.config and 'point_cloud_params' not in processor_config:
            processor_config['point_cloud_params'] = self.config['point_cloud_params']
            del self.config['point_cloud_params']
            print("[INFO] Migrated top-level point_cloud_params -> processor_config.point_cloud_params")
        dataset_paths = processor_config.get('dataset_paths', {})
        raw_path = dataset_paths.get('raw')
        processed_path = dataset_paths.get('processed')
        if not raw_path or not processed_path:
            raise ValueError("processor_config.dataset_paths must contain 'raw' and 'processed' paths")

        # Use single train directory
        train_dir = os.path.join(processed_path, 'train')
        os.makedirs(train_dir, exist_ok=True)

        # Initialize pointcloud processor with output directory (train_dir)
        if self.pointcloud_processor is None:
            self.pointcloud_processor = PointCloudProcessor(
                data_dir=train_dir,
                target_volume=processor_config.get('point_cloud_params', {}).get('target_volume', 6)
            )

        # Discover mesh files
        mesh_files = self._discover_mesh_files(raw_path)
        if not mesh_files:
            raise ValueError(f"No mesh files found in raw path: {raw_path}")
        print(f"    ✓ Found {len(mesh_files)} mesh files")

        # Point sampling params (canonical nested location only)
        pc = processor_config.get('point_cloud_params', {})
        provenance = 'nested' if pc else 'defaults'
        if not pc:
            pc = {'radius': 0.02, 'sigma': 0.0, 'mu': 0.0, 'n_gaussian': 5, 'n_uniform': 1000}
            print("[WARN] Using default point_cloud_params (provide processor_config.point_cloud_params to override)")
        radius = pc.get('radius', 0.02)
        sigma = pc.get('sigma', 0.0)
        mu = pc.get('mu', 0.0)
        n_gaussian = pc.get('n_gaussian', 5)
        n_uniform = pc.get('n_uniform', 1000)
        print(f"    ✓ Using point_cloud_params (source={provenance}): radius={radius} sigma={sigma} n_gaussian={n_gaussian} n_uniform={n_uniform}")

        # Generate (or load existing) point clouds + SDFs
        self.pointcloud_processor.generate_point_cloud(
            meshes=[os.path.join(raw_path, f) for f in mesh_files],
            radius=radius,
            sigma=sigma,
            mu=mu,
            n_gaussian=n_gaussian,
            n_uniform=n_uniform
        )

        # Build dataset_info structure expected by trainer
        z_dim = self.config.get('model_config', {}).get('z_dim', 32)
        latent_mean = trainer_config.get('latent_mean', 0.0)
        latent_sd = trainer_config.get('latent_sd', 0.01)

        train_files = []
        total_points = 0
        processed_files = []
        for mesh_name in [os.path.splitext(f)[0] for f in mesh_files]:
            points_file = os.path.join(train_dir, f"{mesh_name}_sampled_points.npy")
            distances_file = os.path.join(train_dir, f"{mesh_name}_signed_distances.npy")
            if os.path.exists(points_file) and os.path.exists(distances_file):
                processed_files.append(mesh_name)
                try:
                    pts = np.load(points_file, mmap_mode='r')
                    total_points += len(pts)
                except Exception:
                    pass
                train_files.append({
                    'mesh_name': mesh_name,
                    'points_file': points_file,
                    'distances_file': distances_file,
                    'split': 'train'
                })
            else:
                print(f"   Warning: Missing generated data for mesh {mesh_name}")

        dataset_info = {
            'train_files': train_files,
            'dataset_params': {
                'z_dim': z_dim,
                'latent_mean': latent_mean,
                'latent_sd': latent_sd,
                'num_samples': n_uniform + n_gaussian * 10,  # heuristic
                'point_cloud_params': {
                    'radius': radius,
                    'sigma': sigma,
                    'mu': mu,
                    'n_gaussian': n_gaussian,
                    'n_uniform': n_uniform,
                    'source': provenance
                }
            }
        }

        processing_results = {
            'status': 'success',
            'dataset_info': dataset_info,
            'processed_data_path': processed_path,
            'train_files': train_files,
            'processed_files': processed_files,
            'train_count': len(train_files),
            'corrupted_files': [],
            'skipped_files': [],
            'success_rate': (len(processed_files) / len(mesh_files)) if mesh_files else 0,
            'total_points_generated': total_points,
            'processing_params': dataset_info['dataset_params']
        }

        state["processing_report"] = processing_results
        self._log_processing_summary(processing_results)
        self.artifact_manager.save_artifacts(data_processing_results=processing_results)

    def _build_model_step(self, state):
        """\nStep 2: Build or load model architecture."""
        if self._should_skip_step('model_config', 'skip_building'):
            print("Step 2: Model Building - SKIPPED")
            return
        
        print("\nStep 2: Building Model Architecture...")
        
        # Directly construct DeepSDF model from model_config
        model_config = self.config.get('model_config', {})
        model = DeepSDF(model_config)
        state["model"] = model

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    ✓ Model: {model.__class__.__name__}")
        print(f"    ✓ Parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Save model architecture info
        architecture_info = {
            'model_class': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }
        self.artifact_manager.save_artifacts(model_architecture=architecture_info)

    def _train_model_step(self, state):
        """Step 3: Train the model or load pre-trained model."""
        trainer_config = self.config.get('trainer_config', {})

        # Check if we should load a pre-trained model instead of training
        if trainer_config.get('skip_training', False):
            self._load_pretrained_model(state)
            return

        # Train the model
        print("\nStep 3: Training Model...")
        model = state.get("model")
        if not model:
            raise ValueError("No model found. Model building step must be completed first.")

        # Get dataset info if available from data processing step
        processing_report = state.get("processing_report")
        dataset_info = processing_report.get('dataset_info') if processing_report else None

        # Train the model (ModelTrainer will save detailed artifacts automatically)
        training_results = self.model_trainer.run_training(model, dataset_info)

        # Handle different return formats from trainer
        if isinstance(training_results, tuple) and len(training_results) == 2:
            trained_model, training_report = training_results
        else:
            trained_model = training_results
            training_report = {}

        state["trained_model"] = trained_model
        state["training_results"] = training_report

        # Save a basic model copy via artifact manager for compatibility
        if trainer_config.get('save_model', True):
            model_metadata = {
                'total_epochs': training_report.get('total_epochs', 0),
                'final_train_loss': training_report.get('final_train_loss', 0),
                'best_train_loss': training_report.get('best_train_loss', float('inf')),
                'experiment_id': training_report.get('experiment_id'),
                'training_artifacts_location': training_report.get('run_directory')  # Link to detailed artifacts
            }
            model_path = self.artifact_manager.save_model(trained_model, "trained_model", model_metadata)
            if model_path:
                training_report["model_saved_path"] = model_path
                training_report["model_filename"] = os.path.basename(model_path)

        # Save training summary via artifact manager
        self.artifact_manager.save_artifacts(training_results=training_report)
        
        print("\n=== Training Completed Successfully ===\n")

        # Log detailed artifact locations
        if 'training_artifacts' in training_report:
            artifacts = training_report['training_artifacts']
            print(f"Training artifacts saved to: {training_report.get('run_directory', 'N/A')}")
            print(f"    ✓ Best model: {os.path.basename(artifacts.get('best_model_path', 'N/A'))}")
            print(f"    ✓ Final model: {os.path.basename(artifacts.get('final_model_path', 'N/A'))}")
            print(f"    ✓ Latest checkpoint: {os.path.basename(artifacts.get('latest_checkpoint_path', 'N/A'))}")
            print(f"    ✓ Training summary: {os.path.basename(artifacts.get('training_summary_path', 'N/A'))}")
        else:
            print("No artifacts saved.")

    def _load_pretrained_model(self, state):
        """Load a pre-trained model"""
        model_path = self.config.get('evaluator_config', {}).get('model_path')
        if not model_path:
            raise ValueError("No model_path specified in evaluator_config.")
        print(f"Step 3: Loading Pre-trained Model from {model_path}...")
        model = state.get("model")
        if not model:
            raise ValueError("No model architecture available for loading weights.")
        trained_model = self.artifact_manager.load_model(model, model_path)
        state["trained_model"] = trained_model
        print("Pre-trained Model Loaded")

    def _save_pipeline_summary(self, state):
        """Save a summary of the entire pipeline execution."""
        summary = {
            'pipeline_completed': True,
            'steps_completed': [
                'data_processing' if 'processing_report' in state else 'skipped',
                'model_building' if 'model' in state else 'skipped',
                'training' if 'trained_model' in state else 'skipped'
            ],
            'artifacts_generated': list(state.keys())
        }

        if 'processing_report' in state:
            processing_report = state['processing_report']
            summary['data_processing_stats'] = {
                'total_files': len(processing_report.get('processed_files', [])),
                'train_files': processing_report.get('train_count', 0),
                'corrupted_files': len(processing_report.get('corrupted_files', [])),
                'success_rate': processing_report.get('success_rate', 0)
            }

        self.artifact_manager.save_artifacts(pipeline_summary=summary)

        completed_steps = [step for step in summary['steps_completed'] if step != 'skipped']
        print(f"\nPipeline Summary: {', '.join(completed_steps)}")

        if 'data_processing_stats' in summary:
            stats = summary['data_processing_stats']
            if stats['corrupted_files'] > 0:
                print(f"   Data processing: {stats['corrupted_files']} corrupted files skipped")
                print(f"   Success rate: {stats['success_rate']:.1%}")

    def _log_processing_summary(self, results):
        """Log a detailed summary of data processing results."""
        print("\nPoint Cloud Generation Summary:")
        print(f"    ✓ Number of training files: {len(results['train_files'])}")
        print(f"    ✓ Total points generated: {results['total_points_generated']:,}")

        if results.get('corrupted_files'):
            print(f"  Corrupted files skipped: {len(results['corrupted_files'])}")
            print(f"  Success rate: {results.get('success_rate', 0):.1%}")
            if len(results['corrupted_files']) <= 10:
                print(f"  Corrupted files: {', '.join(results['corrupted_files'])}")
            else:
                print(f"  Corrupted files: {', '.join(results['corrupted_files'][:10])}... (and {len(results['corrupted_files'])-10} more)")

    @staticmethod
    def _discover_mesh_files(raw_path):
        """Discover mesh files in raw path (extensions: obj, ply, stl, off, vtk)."""
        if not os.path.exists(raw_path):
            print(f"Warning: Raw path does not exist: {raw_path}")
            return []
        mesh_extensions = {'.obj', '.ply', '.stl', '.off', '.vtk'}
        return [f for f in os.listdir(raw_path) if os.path.isfile(os.path.join(raw_path, f)) and os.path.splitext(f)[1].lower() in mesh_extensions]
