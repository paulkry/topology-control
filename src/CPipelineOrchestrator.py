from src.CDataProcessor import CDataProcessor
from src.CModelTrainer import CModelTrainer
from src.CEvaluator import CEvaluator
from src.CArchitectureManager import CArchitectureManager
import yaml
import os

class CPipelineOrchestrator:
    def __init__(self, config_file="config/config.yaml"):
        self.config = self._load_config(config_file)
        self.architecture_manager = CArchitectureManager(self.config["artifacts_config"])
        self.data_processor = CDataProcessor(self.config["processor_config"])
        self.model_trainer = CModelTrainer(self.config["trainer_config"])
        self.evaluator = CEvaluator(self.config["evaluator_config"], self.config["processor_config"])
    

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

    def run(self):
        pass