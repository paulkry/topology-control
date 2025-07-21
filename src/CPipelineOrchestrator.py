"""
Pipeline orchestrator for 3D shape analysis.

Manages the complete ML pipeline:
- Data processing
- Model building
- Training/Loading
- Evaluation
- Artifact management
"""

from src.CDataProcessor import CDataProcessor
from src.CModelTrainer import CModelTrainer
from src.CEvaluator import CEvaluator
from src.CArchitectureManager import CArchitectureManager
from src.CArtifactManager import CArtifactManager

import yaml
import os
import traceback

class CPipelineOrchestrator:
    def __init__(self, config_file="config/config.yaml"):
        self.config = self._load_config(config_file)
        self._current_step = None
        
        # Initialize artifact manager first to get experiment info
        self.artifact_manager = CArtifactManager(self.config["artifacts_config"])
        
        # Get experiment info for component integration
        experiment_summary = self.artifact_manager.get_experiment_summary()
        experiment_id = experiment_summary['experiment_id']
        artifacts_base = self.artifact_manager.artifacts_path  # FIXED: Use artifacts_path instead of artifacts_base
        
        # Initialize pipeline components with experiment integration
        self.architecture_manager = CArchitectureManager(self.config["model_config"])
        self.data_processor = CDataProcessor(self.config["processor_config"])
        
        # Pass experiment info to trainer for integrated artifact saving
        trainer_config = self.config["trainer_config"].copy()
        trainer_config['experiment_id'] = experiment_id
        trainer_config['artifacts_base'] = artifacts_base
        self.model_trainer = CModelTrainer(trainer_config)
        
        self.evaluator = CEvaluator(self.config["evaluator_config"], self.config["processor_config"])

    def run_full_pipeline(self):
        """Execute the complete ML pipeline: Data → Model → Training → Evaluation."""
        pipeline_state = {}
        
        try:
            self._log_experiment_start()
            
            # Step 1: Data Processing
            self._current_step = 'data_processing'
            self._process_data_step(pipeline_state)
            
            # Step 2: Model Building
            self._current_step = 'model_building'
            self._build_model_step(pipeline_state)
            
            # Step 3: Training (or Loading)
            self._current_step = 'training'
            self._train_model_step(pipeline_state)
            
            # Step 4: Evaluation
            self._current_step = 'evaluation'
            self._evaluate_model_step(pipeline_state)
            
            # Save final summary
            self._save_pipeline_summary(pipeline_state)
            print("\n=== Pipeline Completed Successfully ===")
            
        except Exception as e:
            self._handle_pipeline_error(e)
            raise

    def _load_config(self, config_file):
        with open(config_file, "r") as file:
            print("- Configuration Loaded -")
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
            'trainer_config': ['fine_tune_model_path', 'fine_tune_data_path'],
            'evaluator_config': ['test_data_path', 'model_path']
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

    def _log_experiment_start(self):
        """Log experiment information."""
        print("=" * 50)
        print("🚀 STARTING ML PIPELINE")
        print("=" * 50)
        experiment_summary = self.artifact_manager.get_experiment_summary()
        print(f"📋 Experiment ID: {experiment_summary['experiment_id']}")
        print(f"📁 Artifacts Path: {experiment_summary['path']}")
        print()

    def _handle_pipeline_error(self, error):
        """Handle pipeline errors and save error information."""
        print(f"\n❌ Pipeline FAILED at step '{self._current_step}': {error}")
        error_info = {
            'error': str(error),
            'failed_step': self._current_step,
            'traceback': traceback.format_exc()
        }
        self.artifact_manager.save_artifacts(error_report=error_info)

    def _should_skip_step(self, config_section, skip_key, default=True):
        """Check if a pipeline step should be skipped based on configuration."""
        return self.config.get(config_section, {}).get(skip_key, default)

    def _process_data_step(self, pipeline_state):
        """Process raw data into training format."""
        if self.config['processor_config'].get('skip_processing', False):
            print("⏭️  Skipping data processing (skip_processing=True)")
            return
        
        print("📊 Step 1: Processing Data...")
        
        processor_config = self.config['processor_config']
        processor = CDataProcessor(processor_config)
        
        # Generate dataset for SDF training with volumes
        processing_results = processor.generate_sdf_dataset(
            z_dim=self.config.get('model_config', {}).get('z_dim', 16),
            latent_mean=self.config.get('trainer_config', {}).get('latent_mean', 0.0),
            latent_sd=self.config.get('trainer_config', {}).get('latent_sd', 0.01)
        )
        
        # FIXED: Extract dataset_info from processing_results
        pipeline_state['dataset_info'] = processing_results.get('dataset_info', None)
        
        if pipeline_state['dataset_info'] is None:
            raise ValueError("Failed to generate dataset_info from data processor")
        
        pipeline_state['processed_data_path'] = processing_results.get('processed_data_path')
        pipeline_state['processing_report'] = processing_results  # Store for summary
        
        # Save processing report
        if processing_results:
            self._save_processing_report(processing_results)
        
        print("✅ Step 1: Data Processing Complete")

    def _build_model_step(self, state):
        """Step 2: Build or load model architecture."""
        if self._should_skip_step('model_config', 'skip_building'):
            print("🏗️  Step 2: Model Building - SKIPPED")
            return
        
        print("🏗️  Step 2: Building Model Architecture...")
        model = self.architecture_manager.get_model()
        state["model"] = model
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Model: {model.__class__.__name__}")
        print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Save model architecture info
        architecture_info = {
            'model_class': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }
        
        # Add architecture info to state for summary
        state['architecture_info'] = architecture_info
        
        self.artifact_manager.save_artifacts(model_architecture=architecture_info)
        print("✅ Step 2: Model Building Complete")

    def _train_model_step(self, pipeline_state):
        """Train the model using processed data."""
        if self.config['trainer_config'].get('skip_training', False):
            print("⏭️  Skipping model training (skip_training=True)")
            return
        
        print("🎯 Step 3: Training Model...")
        
        # FIXED: Pass dataset_info to trainer
        dataset_info = pipeline_state.get('dataset_info')
        if dataset_info is None:
            raise ValueError("No dataset_info available for training. Check data processing step.")
        
        model = pipeline_state['model']
        
        # ENHANCED: Handle the return value from the updated CModelTrainer
        try:
            trained_model, training_report = self.model_trainer.train_and_validate(model, dataset_info)
            
            # Extract training metrics from the enhanced training report
            training_results = {
                'model': trained_model,
                'status': 'success',
                'metrics': training_report,
                'final_loss': training_report.get('final_val_loss', 'N/A'),
                'best_loss': training_report.get('best_val_loss', 'N/A'),
                'epochs_completed': training_report.get('total_epochs', 'N/A'),
                'training_time': training_report.get('total_training_time', 'N/A'),
                'model_saved': training_report.get('best_model_saved', False),
                'save_directory': training_report.get('model_save_directory', 'N/A'),
                'device_used': training_report.get('device_used', 'unknown')
            }
            
            # Log training completion with enhanced model file information
            print(f"  ⏱️  Training time: {training_results['training_time']:.2f}s")
            print(f"  🎯 Best validation loss: {training_results['best_loss']:.6f}")
            print(f"  🖥️  Device used: {training_results['device_used']}")
            print(f"  💾 Model saved: {'✓' if training_results['model_saved'] else '✗'}")
            
            if training_results['model_saved']:
                save_dir = training_results['save_directory']
                print(f"  📁 Save directory: {save_dir}")
                
                # List saved PyTorch model files (.pth)
                if os.path.exists(save_dir):
                    model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
                    if model_files:
                        print(f"  📄 Model files ({len(model_files)}): {', '.join(model_files)}")
                        
                        # Highlight the best model file
                        best_files = [f for f in model_files if 'best' in f.lower()]
                        if best_files:
                            print(f"  🏆 Best model: {best_files[0]}")
                    else:
                        print(f"  ⚠️  No .pth files found in save directory")
                else:
                    print(f"  ⚠️  Save directory does not exist: {save_dir}")
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            training_results = {
                'model': model,  # Return original model
                'status': 'failed',
                'error': str(e),
                'final_loss': 'N/A',
                'best_loss': 'N/A',
                'epochs_completed': 0,
                'training_time': 0,
                'model_saved': False,
                'save_directory': 'N/A'
            }
        
        pipeline_state['trained_model'] = training_results['model']
        pipeline_state['training_results'] = training_results
        
        self._save_training_artifacts(training_results)
        print("✅ Step 3: Model Training Complete")
        
    def _evaluate_model_step(self, pipeline_state):
        """Evaluate the trained model."""
        if self.config.get('evaluator_config', {}).get('skip_evaluation', False):
            print("⏭️  Skipping model evaluation (skip_evaluation=True)")
            return
        
        print("📊 Step 4: Evaluating Model...")
        
        # Get the trained model
        trained_model = pipeline_state.get('trained_model')
        if trained_model is None:
            print("⚠️  No trained model available for evaluation. Using original model.")
            trained_model = pipeline_state.get('model')
            if trained_model is None:
                raise ValueError("No model available for evaluation")
        
        # Get dataset info
        dataset_info = pipeline_state.get('dataset_info')
        if dataset_info is None:
            raise ValueError("No dataset_info available for evaluation. Check data processing step.")
        
        try:
            # Get evaluation parameters from config
            eval_config = self.config.get('evaluator_config', {})
            batch_size = eval_config.get('batch_size', 16)
            resolution = eval_config.get('resolution', 50)
            
            # Run evaluation
            evaluation_results = self.evaluator.evaluate(
                model=trained_model,
                dataset_info=dataset_info,
                batch_size=batch_size,
                resolution=resolution
            )
            
            # Log evaluation results
            print(f"  📊 Evaluation complete!")
            print(f"     Model type: {evaluation_results.get('model_type', 'Unknown')}")
            print(f"     Samples evaluated: {evaluation_results.get('num_samples', 'N/A')}")
            print(f"     Average SDF loss: {evaluation_results.get('average_sdf_loss', 'N/A'):.6f}")
            
            if evaluation_results.get('average_volume_loss') is not None:
                print(f"     Average volume loss: {evaluation_results.get('average_volume_loss', 'N/A'):.6f}")
                vol_stats = evaluation_results.get('volume_statistics', {})
                if 'mae' in vol_stats:
                    print(f"     Volume MAE: {vol_stats['mae']:.6f}")
                    print(f"     Volume RMSE: {vol_stats['rmse']:.6f}")
            
            print(f"     Meshes extracted: {len(evaluation_results.get('extracted_meshes', []))}")
            print(f"     Resolution used: {evaluation_results.get('resolution', 'N/A')}")
            
            # Store results in pipeline state
            pipeline_state['evaluation_results'] = evaluation_results
            
            # Save evaluation artifacts
            self._save_evaluation_artifacts(evaluation_results)
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            evaluation_results = {
                'status': 'failed',
                'error': str(e),
                'model_type': trained_model.__class__.__name__ if trained_model else 'Unknown'
            }
            pipeline_state['evaluation_results'] = evaluation_results
            
            # Still save the error results
            self._save_evaluation_artifacts(evaluation_results)
        
        print("✅ Step 4: Model Evaluation Complete")

    def _load_pretrained_model(self, state):
        """Load a pre-trained model for evaluation."""
        model_path = self.config.get('evaluator_config', {}).get('model_path')
        if not model_path:
            raise ValueError("Training is skipped but no model_path specified in evaluator_config.")
        
        print(f"📁 Step 3: Loading Pre-trained Model from {model_path}...")
        model = state.get("model")
        if not model:
            raise ValueError("No model architecture available for loading weights.")
        
        # Use artifact manager to load the model
        trained_model = self.artifact_manager.load_model(model, model_path)
        
        state["trained_model"] = trained_model
        print("✅ Step 3: Pre-trained Model Loaded")

    def _save_pipeline_summary(self, state):
        """Save a comprehensive summary of the entire pipeline execution."""
        # Get final artifact locations
        training_results = state.get("training_results", {})
        evaluation_results = state.get("evaluation_results", {})
        
        summary = {
            'pipeline_completed': True,
            'steps_completed': [
                'data_processing' if 'processing_report' in state else 'skipped',
                'model_building' if 'model' in state else 'skipped', 
                'training' if 'trained_model' in state else 'skipped',
                'evaluation' if 'evaluation_results' in state else 'skipped'
            ],
            'artifacts_generated': list(state.keys()),
            'experiment_id': getattr(self.artifact_manager, 'experiment_id', 'unknown')
        }
        
        # Add data processing stats to summary
        if 'processing_report' in state:
            processing_report = state['processing_report']
            dataset_info = processing_report.get('dataset_info', {})
            summary['data_processing_stats'] = {
                'total_files': len(processing_report.get('train_files', [])) + len(processing_report.get('val_files', [])),
                'train_files': len(processing_report.get('train_files', [])),
                'val_files': len(processing_report.get('val_files', [])),
                'corrupted_files': len(processing_report.get('corrupted_files', [])),
                'success_rate': processing_report.get('success_rate', 0),
                'z_dim': dataset_info.get('dataset_params', {}).get('z_dim', 'unknown')
            }
        
        # Add training stats to summary with model file information
        if 'training_results' in state:
            training_results = state['training_results']
            training_stats = {
                'status': training_results.get('status', 'unknown'),
                'final_loss': training_results.get('final_loss', 'N/A'),
                'best_loss': training_results.get('best_loss', 'N/A'),
                'epochs_completed': training_results.get('epochs_completed', 'N/A'),
                'training_time': training_results.get('training_time', 'N/A'),
                'model_saved': training_results.get('model_saved', False),
                'save_directory': training_results.get('save_directory', 'N/A'),
                'device_used': training_results.get('device_used', 'unknown')
            }
            
            # ENHANCED: Add model file information to summary
            if training_results.get('model_saved'):
                save_dir = training_results.get('save_directory', 'N/A')
                if save_dir != 'N/A' and os.path.exists(save_dir):
                    model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
                    training_stats['model_files'] = model_files
                    training_stats['model_file_count'] = len(model_files)
                    
                    # Calculate total model storage size
                    total_size_mb = 0
                    for file in model_files:
                        filepath = os.path.join(save_dir, file)
                        if os.path.exists(filepath):
                            total_size_mb += os.path.getsize(filepath) / (1024 * 1024)
                    training_stats['total_model_size_mb'] = round(total_size_mb, 2)
                    
                    # Identify best model
                    best_files = [f for f in model_files if 'best' in f.lower()]
                    if best_files:
                        training_stats['best_model_file'] = best_files[0]
                        best_path = os.path.join(save_dir, best_files[0])
                        if os.path.exists(best_path):
                            best_size_mb = os.path.getsize(best_path) / (1024 * 1024)
                            training_stats['best_model_size_mb'] = round(best_size_mb, 2)
            
            summary['training_stats'] = training_stats
        
        # Add evaluation stats if available
        if 'evaluation_results' in state:
            eval_results = state['evaluation_results']
            eval_stats = {
                'status': eval_results.get('status', 'unknown'),
                'model_type': eval_results.get('model_type', 'Unknown'),
                'samples_evaluated': eval_results.get('num_samples', 'N/A'),
                'resolution': eval_results.get('resolution', 'N/A'),
                'meshes_extracted': len(eval_results.get('extracted_meshes', []))
            }
            
            if eval_results.get('status') != 'failed':
                eval_stats['average_sdf_loss'] = eval_results.get('average_sdf_loss', 'N/A')
                if eval_results.get('average_volume_loss') is not None:
                    eval_stats['average_volume_loss'] = eval_results.get('average_volume_loss', 'N/A')
                    vol_stats = eval_results.get('volume_statistics', {})
                    if vol_stats:
                        eval_stats['volume_mae'] = vol_stats.get('mae', 'N/A')
                        eval_stats['volume_rmse'] = vol_stats.get('rmse', 'N/A')
            else:
                eval_stats['error'] = eval_results.get('error', 'Unknown error')
            
            summary['evaluation_stats'] = eval_stats
        
        self.artifact_manager.save_artifacts(pipeline_summary=summary)
        
        # Enhanced summary logging with model and evaluation information
        completed_steps = [step for step in summary['steps_completed'] if step != 'skipped']
        print(f"📋 Pipeline Summary: {', '.join(completed_steps)}")
        
        # Log key statistics
        if 'data_processing_stats' in summary:
            stats = summary['data_processing_stats']
            print(f"   📊 Data: {stats['train_files']} train, {stats['val_files']} val files")
            if stats['corrupted_files'] > 0:
                print(f"   ⚠️  {stats['corrupted_files']} corrupted files skipped (success rate: {stats['success_rate']:.1%})")
        
        if 'training_stats' in summary:
            train_stats = summary['training_stats']
            if train_stats['status'] == 'success':
                print(f"   🎯 Training: {train_stats['epochs_completed']} epochs, best loss: {train_stats['best_loss']}")
                if train_stats['model_saved']:
                    model_count = train_stats.get('model_file_count', 0)
                    total_size = train_stats.get('total_model_size_mb', 0)
                    print(f"   💾 Models saved: {model_count} files ({total_size:.1f} MB)")
                    
                    if 'best_model_file' in train_stats:
                        best_file = train_stats['best_model_file']
                        best_size = train_stats.get('best_model_size_mb', 0)
                        print(f"   🏆 Best model: {best_file} ({best_size:.1f} MB)")
                else:
                    print(f"   ❌ No model files saved")
            else:
                print(f"   ❌ Training failed: {train_stats.get('error', 'unknown error')}")
        
        if 'evaluation_stats' in summary:
            eval_stats = summary['evaluation_stats']
            if eval_stats['status'] != 'failed':
                print(f"   📊 Evaluation: {eval_stats['samples_evaluated']} samples, {eval_stats['meshes_extracted']} meshes extracted")
                print(f"   📏 Resolution: {eval_stats['resolution']}, SDF loss: {eval_stats.get('average_sdf_loss', 'N/A')}")
                if 'volume_mae' in eval_stats:
                    print(f"   🧊 Volume MAE: {eval_stats['volume_mae']:.6f}, RMSE: {eval_stats['volume_rmse']:.6f}")
            else:
                print(f"   ❌ Evaluation failed: {eval_stats.get('error', 'unknown error')}")

    def _log_processing_summary(self, results):
        """Log a detailed summary of data processing results."""
        print("\n📊 Data Processing Summary:")
        print(f"  Train files: {len(results.get('train_files', []))}")
        print(f"  Val files: {len(results.get('val_files', []))}")
        print(f"  Total points generated: {results.get('total_points_generated', 0):,}")
        
        if results.get('corrupted_files'):
            print(f"  ⚠️  Corrupted files skipped: {len(results['corrupted_files'])}")
            print(f"  Success rate: {results.get('success_rate', 0):.1%}")
            
            # List corrupted files if there are any
            corrupted_files = results['corrupted_files']
            if len(corrupted_files) <= 10:
                print(f"  Corrupted files: {', '.join(corrupted_files)}")
            else:
                print(f"  Corrupted files: {', '.join(corrupted_files[:10])}... (and {len(corrupted_files)-10} more)")
        
        print()

    def _save_processing_report(self, processing_results):
        """Save data processing results as artifacts."""
        # Log processing summary
        self._log_processing_summary(processing_results)
        
        # Save processing results as artifact
        self.artifact_manager.save_artifacts(data_processing_results=processing_results)
        
        # Store in pipeline state for summary
        return processing_results

    def _save_training_artifacts(self, training_results):
        """Save training results as artifacts with enhanced model file information."""
        # Create comprehensive training summary
        training_summary = {
            'status': training_results.get('status', 'unknown'),
            'final_loss': training_results.get('final_loss', 'N/A'),
            'best_loss': training_results.get('best_loss', 'N/A'),
            'epochs_completed': training_results.get('epochs_completed', 'N/A'),
            'training_time': training_results.get('training_time', 'N/A'),
            'model_saved': training_results.get('model_saved', False),
            'save_directory': training_results.get('save_directory', 'N/A'),
            'device_used': training_results.get('device_used', 'unknown')
        }
        
        # Add error information if training failed
        if training_results.get('error'):
            training_summary['error'] = training_results['error']
        
        # Add detailed metrics if available
        if 'metrics' in training_results and isinstance(training_results['metrics'], dict):
            training_summary['detailed_metrics'] = training_results['metrics']
        
        # ENHANCED: Add model file information
        if training_results.get('model_saved') and training_results.get('save_directory') != 'N/A':
            save_dir = training_results.get('save_directory')
            if os.path.exists(save_dir):
                # List all PyTorch model files
                model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
                training_summary['saved_model_files'] = model_files
                training_summary['model_file_count'] = len(model_files)
                
                # Identify the best model file
                best_files = [f for f in model_files if 'best' in f.lower()]
                if best_files:
                    training_summary['best_model_file'] = best_files[0]
                    training_summary['best_model_path'] = os.path.join(save_dir, best_files[0])
                
                # Get file sizes for reference
                file_info = []
                for file in model_files:
                    filepath = os.path.join(save_dir, file)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    file_info.append({'name': file, 'size_mb': round(size_mb, 2)})
                training_summary['model_file_details'] = file_info
        
        # Save training artifacts
        self.artifact_manager.save_artifacts(training_results=training_summary)
        
        print(f"  📄 Saved artifact: training_results.txt")

    def _save_evaluation_artifacts(self, evaluation_results):
        """Save evaluation results as artifacts."""
        # Create evaluation summary
        eval_summary = {
            'status': evaluation_results.get('status', 'unknown'),
            'model_type': evaluation_results.get('model_type', 'Unknown'),
            'evaluation_type': evaluation_results.get('evaluation_type', 'Unknown'),
            'num_samples': evaluation_results.get('num_samples', 'N/A'),
            'resolution': evaluation_results.get('resolution', 'N/A'),
            'actual_grid_size': evaluation_results.get('actual_grid_size', 'N/A'),
            'device': evaluation_results.get('device', 'unknown')
        }
        
        # Add error information if evaluation failed
        if evaluation_results.get('error'):
            eval_summary['error'] = evaluation_results['error']
        else:
            # Add loss metrics
            eval_summary['average_sdf_loss'] = evaluation_results.get('average_sdf_loss', 'N/A')
            eval_summary['average_test_loss'] = evaluation_results.get('average_test_loss', 'N/A')
            
            if evaluation_results.get('average_volume_loss') is not None:
                eval_summary['average_volume_loss'] = evaluation_results.get('average_volume_loss', 'N/A')
                vol_stats = evaluation_results.get('volume_statistics', {})
                if vol_stats:
                    eval_summary['volume_mae'] = vol_stats.get('mae', 'N/A')
                    eval_summary['volume_rmse'] = vol_stats.get('rmse', 'N/A')
            
            # Add mesh extraction statistics
            extracted_meshes = evaluation_results.get('extracted_meshes', [])
            eval_summary['meshes_extracted'] = len(extracted_meshes)
            
            if extracted_meshes:
                # Mesh statistics
                total_vertices = sum(mesh['num_vertices'] for mesh in extracted_meshes)
                total_faces = sum(mesh['num_faces'] for mesh in extracted_meshes)
                eval_summary['total_vertices'] = total_vertices
                eval_summary['total_faces'] = total_faces
                eval_summary['avg_vertices_per_mesh'] = total_vertices / len(extracted_meshes)
                eval_summary['avg_faces_per_mesh'] = total_faces / len(extracted_meshes)
            
            # Add SDF statistics
            sdf_stats = evaluation_results.get('sdf_statistics', {})
            if sdf_stats.get('min_values'):
                eval_summary['sdf_range_avg_min'] = sdf_stats.get('avg_min', 'N/A')
                eval_summary['sdf_range_avg_max'] = sdf_stats.get('avg_max', 'N/A')
        
        # Save evaluation artifacts
        self.artifact_manager.save_artifacts(evaluation_results=eval_summary)
        
        print(f"  📄 Saved artifact: evaluation_results.txt")

# Additional utility function for loading models
def load_model_from_checkpoint(checkpoint_path, device='auto'):
    """
    Utility function to load a trained model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model on ('auto', 'cpu', 'cuda')
    
    Returns:
        tuple: (model, latent_vectors, checkpoint_info)
    """
    import torch
    from src.CArchitectureManager import VolumeDeepSDF, DeepSDF
    
    # Load checkpoint
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    architecture_info = checkpoint.get('architecture_info', {})
    model_arch = checkpoint.get('model_architecture', 'VolumeDeepSDF')
    
    # Create model configuration
    model_config = {
        'z_dim': architecture_info.get('z_dim', 16),
        'layer_size': architecture_info.get('layer_size', 256),
        'coord_dim': architecture_info.get('coord_dim', 3),
        'dropout_p': architecture_info.get('dropout_p', 0.2)
    }
    
    if model_arch == 'VolumeDeepSDF':
        model_config['volume_hidden_dim'] = architecture_info.get('volume_hidden_dim', 128)
        model = VolumeDeepSDF(config=model_config)
    elif model_arch == 'DeepSDF':
        model = DeepSDF(config=model_config)
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load latent vectors if available
    latent_vectors = checkpoint.get('latent_vectors', None)
    
    # Create checkpoint info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'train_loss': checkpoint.get('train_loss', 'Unknown'),
        'val_loss': checkpoint.get('val_loss', 'Unknown'),
        'timestamp': checkpoint.get('timestamp', 'Unknown'),
        'device': checkpoint.get('device', 'Unknown'),
        'architecture': model_arch,
        'config': model_config
    }
    
    print(f"✅ Loaded {model_arch} model from checkpoint")
    print(f"   Epoch: {checkpoint_info['epoch']}")
    print(f"   Validation loss: {checkpoint_info['val_loss']}")
    if latent_vectors is not None:
        print(f"   Latent vectors: {latent_vectors.shape}")
    
    return model, latent_vectors, checkpoint_info