"""
   [To be Completed] Classifier training pipeline
"""
class CModelTrainer:   
    def __init__(self, functions, loss_fn, x, lr, log_freq=100, verbose=False):
        """
        Initialize the trainer with a list of functions, a loss function, and an initial parameter value
        
        Parameters:
            To be Defined
        """
        self.functions = functions
        self.loss_fn = loss_fn
        self.optimizer = None
        self.log_freq = log_freq
        self.verbose = verbose

    def train(self, target, epochs=1000):
        
        return None