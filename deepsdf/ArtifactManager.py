import os
import torch
import datetime
import json
import base64

class ArtifactManager:
    def __init__(self, config):
        """
        Initialize the artifact manager with configuration parameters.
        
        Parameters:
            config (dict): Configuration parameters for artifact management including:
                - save_artifacts_to: path where artifacts will be saved
        """
        self.config = config
        
        # Extract artifacts path from config
        self.artifacts_path = config.get('save_artifacts_to', 'artifacts')
        
        # Create artifacts directory if it doesn't exist
        os.makedirs(self.artifacts_path, exist_ok=True)
        
        # Generate experiment ID based on timestamp
        self.experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_path = os.path.join(self.artifacts_path, f"experiment_{self.experiment_id}")
        
        # Create experiment directory
        os.makedirs(self.experiment_path, exist_ok=True)
        
        print(f"    ✓ Experiment ID: {self.experiment_id}")
        print(f"    ✓ Experiment will be saved to: {self.experiment_path}")

    def get_experiment_summary(self):
        """
        Get summary information about the current experiment.
        
        Returns:
            dict: Experiment summary with ID and path
        """
        return {
            'experiment_id': self.experiment_id,
            'path': self.experiment_path,
            'artifacts_directory': self.artifacts_path
        }

    def save_artifacts(self, **artifacts):
        """
        Save multiple artifacts (text files and images).
        
        Parameters:
            **artifacts: Key-value pairs where key is the artifact name and value is the data
        """
        saved_files = []
        
        for artifact_name, artifact_data in artifacts.items():
            # Check if this artifact contains base64 image data
            if self._is_base64_image(artifact_name, artifact_data):
                filename = self._save_image_artifact(artifact_name, artifact_data)
                if filename:
                    saved_files.append(filename)
            elif isinstance(artifact_data, dict):
                # For dictionary artifacts, check each key for potential images
                for key, value in artifact_data.items():
                    if self._is_base64_image(key, value):
                        # Save image separately
                        image_filename = self._save_image_artifact(f"{artifact_name}_{key}", value)
                        if image_filename:
                            saved_files.append(image_filename)
                
                filename = self._save_single_artifact(artifact_name, artifact_data)
                if filename:
                    saved_files.append(filename)
            else:
                filename = self._save_single_artifact(artifact_name, artifact_data)
                if filename:
                    saved_files.append(filename)
        
        return saved_files

    def _is_base64_image(self, name, data):
        """
        Check if the data is a base64 encoded image.
        
        Parameters:
            name (str): Name of the artifact
            data: Data to check
            
        Returns:
            bool: True if data appears to be base64 image data
        """
        # Check if it's a plot or image artifact with base64 string data
        if isinstance(data, str) and ('plot' in name.lower() or 'image' in name.lower()):
            try:
                # Try to decode as base64 - if successful and reasonable length, it's likely an image
                decoded = base64.b64decode(data)
                return len(decoded) > 100  # Reasonable minimum size for an image
            except Exception:
                return False
        return False

    def _save_image_artifact(self, name, base64_data):
        """
        Save a base64 encoded image as a PNG file.
        
        Parameters:
            name (str): Name of the artifact
            base64_data (str): Base64 encoded image data
            
        Returns:
            str: Filename of the saved image
        """
        try:
            # Create safe filename
            safe_name = self._sanitize_filename(name)
            filename = f"{safe_name}.png"
            filepath = os.path.join(self.experiment_path, filename)
            
            # Decode base64 and save as PNG
            image_data = base64.b64decode(base64_data)
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            print(f"    ✓ Saved image: {filename}")
            return filename
            
        except Exception as e:
            print(f"Failed to save image {name}: {e}")
            # Fallback to text file
            return self._save_single_artifact(name, f"Base64 image data (failed to decode): {str(e)}")

    def _save_single_artifact(self, name, data):
        """
        Save a single artifact to a text file.
        
        Parameters:
            name (str): Name of the artifact
            data: Data to save (will be converted to text)
            
        Returns:
            str: Filename of the saved artifact
        """
        # Create safe filename
        safe_name = self._sanitize_filename(name)
        filename = f"{safe_name}.txt"
        filepath = os.path.join(self.experiment_path, filename)

        # Convert data to text format
        text_content = self._convert_to_text(name, data)

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text_content)

        print(f"    ✓ Saved artifact: {filename}")
        return filename

    def _convert_to_text(self, name, data):
        """
        Convert various data types to text format.
        
        Parameters:
            name (str): Name of the artifact
            data: Data to convert
            
        Returns:
            str: Text representation of the data
        """
        # Add header with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"Artifact: {name}\nGenerated: {timestamp}\nExperiment: {self.experiment_id}\n"
        header += "=" * 50 + "\n\n"
        
        try:
            # Handle different data types
            if isinstance(data, dict):
                # Exclude base64 image data from text representation
                filtered_data = {k: v for k, v in data.items() if not self._is_base64_image(k, v)}
                content = json.dumps(filtered_data, indent=2, default=str)
            elif isinstance(data, (list, tuple)):
                content = "\n".join(str(item) for item in data)
            elif hasattr(data, '__dict__'):  # Objects with attributes
                content = f"Object Type: {type(data).__name__}\n"
                content += json.dumps(data.__dict__, indent=2, default=str)
            else:
                content = str(data)
        except Exception as e:
            content = f"Error converting data: {e}\nRaw data: {repr(data)}"
        
        return header + content

    def _sanitize_filename(self, filename):
        """
        Sanitize filename by removing/replacing invalid characters.
        
        Parameters:
            filename (str): Original filename
            
        Returns:
            str: Sanitized filename
        """
        # Replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Remove multiple consecutive underscores
        while '__' in filename:
            filename = filename.replace('__', '_')
        
        # Trim underscores from start and end
        filename = filename.strip('_')
        
        # Ensure filename is not empty
        if not filename:
            filename = 'unnamed_artifact'
        
        return filename
    
    def get_artifact_path(self, artifact_name):
        """
        Get the full path to a specific artifact.
        
        Parameters:
            artifact_name (str): Name of the artifact
            
        Returns:
            str: Full path to the artifact file
        """
        safe_name = self._sanitize_filename(artifact_name)
        if not safe_name.endswith('.txt'):
            safe_name += '.txt'
        
        return os.path.join(self.experiment_path, safe_name)

    def save_model(self, model, name="model", metadata=None):
        """
        Save a PyTorch model as a .pt file.
        
        Parameters:
            model: PyTorch model to save
            name (str): Name for the model file (without extension)
            metadata (dict): Optional metadata about the model (will be saved separately)
            
        Returns:
            str: Full path to the saved model file
        """
        try:
            # Generate filename with timestamp only
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Add best training loss to filename if provided
            if metadata and 'best_train_loss' in metadata:
                train_loss = metadata['best_train_loss']
                filename = f"{name}_{timestamp}_train{train_loss:.6f}.pt"
            else:
                filename = f"{name}_{timestamp}.pt"
            
            # Sanitize filename
            safe_filename = self._sanitize_filename(filename.replace('.pt', '')) + '.pt'
            filepath = os.path.join(self.experiment_path, safe_filename)
            
            # Save model state dict
            torch.save(model.state_dict(), filepath)
            print(f"    ✓ Model saved: {safe_filename}")
            
            # Save model metadata if provided
            if metadata:
                metadata_filename = safe_filename.replace('.pt', '_metadata.txt')
                self._save_single_artifact(metadata_filename.replace('.txt', ''), metadata)
            
            return filepath
            
        except Exception as e:
            print(f"Failed to save model {name}: {e}")
            return None
    
    def load_model(self, model, model_path, device=None):
        """
        Load a PyTorch model from a .pt file.
        
        Parameters:
            model: PyTorch model architecture (must match saved model)
            model_path (str): Path to the .pt file (can be absolute or relative to experiment path)
            device: PyTorch device to load model to
            
        Returns:
            model: Loaded model
        """
        try:
            # Handle relative paths
            if not os.path.isabs(model_path):
                model_path = os.path.join(self.experiment_path, model_path)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Determine device
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model state dict
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            
            print(f"Model loaded from: {os.path.basename(model_path)}")
            return model
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
