"""
 [To be Completed] Evaluate the classifier model on the given point cloud.
"""

class CEvaluator:
    def __init__(self, model, device='cpu'):
        """
        Initialize the trainer with a list of functions, a loss function, and an initial parameter value
        
        Parameters:
            To be Defined
        """
        self.model = model
        self.device = device

    def evaluate(self, point_cloud):
        """
        Evaluate the classifier on the provided point cloud.
        """
        self.model.eval()
        
        return None