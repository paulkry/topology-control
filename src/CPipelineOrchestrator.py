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
        
        # Initialize pipeline components
        self.architecture_manager = CArchitectureManager(self.config["model_config"])
        self.data_processor = CDataProcessor(self.config["processor_config"])
        self.model_trainer = CModelTrainer(self.config["trainer_config"])
        self.evaluator = CEvaluator(self.config["evaluator_config"], self.config["processor_config"])
        self.artifact_manager = CArtifactManager(self.config["artifacts_config"])


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
            # self._current_step = 'evaluation'
            # self._evaluate_model_step(pipeline_state)
            
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
        processing_results = self.data_processor.process()
        state["processing_report"] = processing_results
        
        if processing_results:
            self.artifact_manager.save_artifacts(
                data_processing_results=processing_results
            )
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
        
        # Train the model
        training_results = self.model_trainer.train_and_validate(model)
        
        # Handle different return formats from trainer
        if isinstance(training_results, tuple) and len(training_results) == 2:
            trained_model, training_report = training_results
        else:
            trained_model = training_results
            training_report = {}
        
        state["trained_model"] = trained_model
        state["training_results"] = training_report
        
        # Save the trained model using artifact manager
        trainer_config = self.config.get('trainer_config', {})
        if trainer_config.get('save_model', True):
            model_metadata = {
                'best_val_loss': training_report.get('best_val_loss', float('inf')),
                'total_epochs': training_report.get('total_epochs', 0),
                'final_train_loss': training_report.get('final_train_loss', 0),
                'final_val_loss': training_report.get('final_val_loss', 0)
            }
            model_path = self.artifact_manager.save_model(trained_model, "trained_model", model_metadata)
            if model_path:
                training_report["model_saved_path"] = model_path
                training_report["model_filename"] = os.path.basename(model_path)
        
        # Save training artifacts
        self.artifact_manager.save_artifacts(
            training_results=training_report
        )
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
        
        # Run evaluation
        evaluation_results = self.evaluator.evaluate(trained_model)
        state["evaluation_results"] = evaluation_results
        
        # Save evaluation results
        self.artifact_manager.save_artifacts(
            evaluation_results=evaluation_results
        )
        print("✅ Step 4: Model Evaluation Complete")

    def _save_pipeline_summary(self, state):
        """Save a summary of the entire pipeline execution."""
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
        self.artifact_manager.save_artifacts(pipeline_summary=summary)
        print(f"📋 Pipeline Summary: {', '.join(summary['steps_completed'])}")
