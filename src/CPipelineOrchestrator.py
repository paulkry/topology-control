from src.CDataProcessor import CDataProcessor
from src.CArchitectureManager import CArchitectureManager
from src.CModelTrainer import CModelTrainer
from src.evaluator.CEvaluator import CEvaluator

class CPipelineOrchestrator:
    def __init__(self, config_file="config/config.yaml"):
        self.config = self._load_config(config_file)
        
        self.architecture_manager = CArchitectureManager(self.config["model_config"])
        self.data_processor = CDataProcessor(self.config["processor_config"])
        self.model_trainer = CModelTrainer(self.config["trainer_config"])
        self.evaluator = CEvaluator(self.config["evaluator_config"], self.config["processor_config"])

