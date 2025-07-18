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
    def __init__(self, config_file="config/config_examples.yaml"):
        self.config = self._load_config(config_file)
        self._current_step = None
        
        # Initialize artifact manager first to get experiment info
        self.artifact_manager = CArtifactManager(self.config["artifacts_config"])
        
        # Get experiment info for component integration
        experiment_summary = self.artifact_manager.get_experiment_summary()
        experiment_id = experiment_summary['experiment_id']
        artifacts_base = self.config["artifacts_config"]["save_artifacts_to"]
        
        # Initialize pipeline components with experiment integration
        self.architecture_manager = CArchitectureManager(self.config["model_config"])
        self.data_processor = CDataProcessor(self.config["processor_config"])
        
        # Pass experiment info to trainer for integrated artifact saving
        trainer_config = self.config["trainer_config"].copy()
        trainer_config['experiment_id'] = experiment_id
        trainer_config['artifacts_base'] = artifacts_base
        self.model_trainer = CModelTrainer(trainer_config)
        
        self.evaluator = CEvaluator(self.config["evaluator_config"], self.config["processor_config"])

    def run(self):
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
            
            # # Step 4: Evaluation
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

    def _process_data_step(self, state):
        """Step 1: Process and prepare data for training."""
        if self._should_skip_step('processor_config', 'skip_processing'):
            print("🔄 Step 1: Data Processing - SKIPPED")
            return
        
        print("🔄 Step 1: Processing Data...")
        
        # Check if we're using SDF dataset type from trainer config
        trainer_config = self.config.get('trainer_config', {})
        dataset_type = trainer_config.get('dataset_type', 'shape')
        
        if dataset_type == 'sdf':
            # For SDF datasets, generate SDF dataset with proper dataset_info
            print("   Generating SDF dataset...")
            
            # Get SDF parameters from config
            z_dim = self.config.get('model_config', {}).get('z_dim', 32)
            latent_mean = trainer_config.get('latent_mean', 0.0)
            latent_sd = trainer_config.get('latent_sd', 0.01)
            
            processing_results = self.data_processor.generate_sdf_dataset(
                z_dim=z_dim,
                latent_mean=latent_mean,
                latent_sd=latent_sd
            )
        else:
            # For traditional datasets, use standard processing
            print("   Processing traditional dataset...")
            processing_results = self.data_processor.process()
        
        state["processing_report"] = processing_results
        
        if processing_results:
            self._log_processing_summary(processing_results['processing_results'])
        
            self.artifact_manager.save_artifacts(
                data_processing_results=processing_results
            )
        
        # Debug: Check if dataset_info is available
        if isinstance(processing_results, dict) and 'dataset_info' in processing_results:
            print(f"   ✅ Dataset info generated with {len(processing_results['dataset_info']['train_files'])} train files")
            print(f"   ✅ Dataset info generated with {len(processing_results['dataset_info']['val_files'])} val files")
        else:
            print("   ⚠️  No dataset_info in processing results")
        
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
        self.artifact_manager.save_artifacts(model_architecture=architecture_info)
        print("✅ Step 2: Model Building Complete")

    def _train_model_step(self, state):
        """Step 3: Train the model or load pre-trained model."""
        trainer_config = self.config.get('trainer_config', {})
        
        # Check if we should load a pre-trained model instead of training
        if trainer_config.get('skip_training', False):
            self._load_pretrained_model(state)
            return
        
        # Train the model
        print("🎯 Step 3: Training Model...")
        model = state.get("model")
        if not model:
            raise ValueError("No model found. Model building step must be completed first.")
        
        # Get dataset info if available from data processing step
        processing_report = state.get("processing_report")
        dataset_info = processing_report.get('dataset_info')
        
        # Train the model (CModelTrainer will save detailed artifacts automatically)
        training_results = self.model_trainer.train_and_validate(model, dataset_info)
        
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
                'best_val_loss': training_report.get('best_val_loss', float('inf')),
                'total_epochs': training_report.get('total_epochs', 0),
                'final_train_loss': training_report.get('final_train_loss', 0),
                'final_val_loss': training_report.get('final_val_loss', 0),
                'experiment_id': training_report.get('experiment_id'),
                'training_artifacts_location': training_report.get('run_directory')  # Link to detailed artifacts
            }
            model_path = self.artifact_manager.save_model(trained_model, "trained_model", model_metadata)
            if model_path:
                training_report["model_saved_path"] = model_path
                training_report["model_filename"] = os.path.basename(model_path)
        
        # Save training summary via artifact manager
        self.artifact_manager.save_artifacts(
            training_results=training_report
        )
        
        # Log detailed artifact locations
        if 'training_artifacts' in training_report:
            artifacts = training_report['training_artifacts']
            print("✅ Step 3: Model Training Complete")
            print(f"📁 Training artifacts saved to: {training_report.get('run_directory', 'N/A')}")
            print(f"   📄 Best model: {os.path.basename(artifacts.get('best_model_path', 'N/A'))}")
            print(f"   📄 Final model: {os.path.basename(artifacts.get('final_model_path', 'N/A'))}")
            print(f"   📄 Latest checkpoint: {os.path.basename(artifacts.get('latest_checkpoint_path', 'N/A'))}")
            print(f"   📄 Training summary: {os.path.basename(artifacts.get('training_summary_path', 'N/A'))}")
        else:
            print("✅ Step 3: Model Training Complete")

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

    def _evaluate_model_step(self, state):
        """Step 4: Evaluate the trained model."""
        if self._should_skip_step('evaluator_config', 'skip_evaluation', default=False):
            print("📊 Step 4: Model Evaluation - SKIPPED")
            return
        
        print("📊 Step 4: Evaluating Model...")
        trained_model = state.get("trained_model")
        if not trained_model:
            raise ValueError("No trained model found. Training step must be completed first.")
        
        # Get dataset info for evaluation if available
        processing_report = state.get("processing_report")
        dataset_info = None
        if processing_report and isinstance(processing_report, dict):
            dataset_info = processing_report.get('dataset_info')
        
        # Run evaluation
        evaluation_results = self.evaluator.evaluate_sdf_dataset(trained_model, dataset_info)
        state["evaluation_results"] = evaluation_results
        
        # Save evaluation results
        self.artifact_manager.save_artifacts(
            evaluation_results=evaluation_results
        )
        print("✅ Step 4: Model Evaluation Complete")

    def _save_pipeline_summary(self, state):
        """Save a summary of the entire pipeline execution."""
        # Get final artifact locations
        training_results = state.get("training_results", {})
        artifact_locations = {}
        
        if 'training_artifacts' in training_results:
            artifact_locations = training_results['training_artifacts']
        
        summary = {
            'pipeline_completed': True,
            'steps_completed': [
                'data_processing' if 'processing_report' in state else 'skipped',
                'model_building' if 'model' in state else 'skipped', 
                'training' if 'trained_model' in state else 'skipped',
                'evaluation' if 'evaluation_results' in state else 'skipped'
            ],
            'artifacts_generated': list(state.keys())
        }
        
        # Add data processing stats to summary
        if 'processing_report' in state:
            processing_report = state['processing_report']
            summary['data_processing_stats'] = {
                'total_files': len(processing_report.get('processed_files', [])),
                'train_files': processing_report.get('train_count', 0),
                'val_files': processing_report.get('val_count', 0),
                'corrupted_files': len(processing_report.get('corrupted_files', [])),
                'success_rate': processing_report.get('success_rate', 0)
            }
        
        self.artifact_manager.save_artifacts(pipeline_summary=summary)
        
        # Enhanced summary logging
        completed_steps = [step for step in summary['steps_completed'] if step != 'skipped']
        print(f"📋 Pipeline Summary: {', '.join(completed_steps)}")
        
        if 'data_processing_stats' in summary:
            stats = summary['data_processing_stats']
            if stats['corrupted_files'] > 0:
                print(f"   ⚠️  Data processing: {stats['corrupted_files']} corrupted files skipped")
                print(f"   Success rate: {stats['success_rate']:.1%}")

    def _log_processing_summary(self, results):
        """Log a detailed summary of data processing results."""
        print("results", results)
        print("\n📊 Data Processing Summary:")
        print(f"  Total files processed: {len(results['processed_files'])}")
        print(f"  Train files: {results['train_count']}")
        print(f"  Val files: {results['val_count']}")
        print(f"  Total points generated: {results['total_points_generated']:,}")
        
        if results.get('corrupted_files'):
            print(f"  ⚠️  Corrupted files skipped: {len(results['corrupted_files'])}")
            print(f"  Success rate: {results.get('success_rate', 0):.1%}")
            
            # List corrupted files if there are any
            if len(results['corrupted_files']) <= 10:
                print(f"  Corrupted files: {', '.join(results['corrupted_files'])}")
            else:
                print(f"  Corrupted files: {', '.join(results['corrupted_files'][:10])}... (and {len(results['corrupted_files'])-10} more)")
        
        print()
