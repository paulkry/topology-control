"""
Artifact Management for Topology Control Pipeline
"""
import os
import json
import datetime
from pathlib import Path

class CArtifactManager:
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
        
        print(f"Artifact Manager initialized - Experiment: {self.experiment_id}")
        print(f"Artifacts will be saved to: {self.experiment_path}")

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
        Save multiple artifacts as text files.
        
        Parameters:
            **artifacts: Key-value pairs where key is the artifact name and value is the data
        """
        saved_files = []
        
        for artifact_name, artifact_data in artifacts.items():
            try:
                filename = self._save_single_artifact(artifact_name, artifact_data)
                saved_files.append(filename)
                print(f"  📄 Saved: {filename}")
            except Exception as e:
                print(f"  ❌ Failed to save {artifact_name}: {e}")
        
        return saved_files

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
                content = json.dumps(data, indent=2, default=str)
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

    def save_text_file(self, filename, content):
        """
        Save arbitrary text content to a file.
        
        Parameters:
            filename (str): Name of the file
            content (str): Text content to save
            
        Returns:
            str: Full path to the saved file
        """
        safe_filename = self._sanitize_filename(filename)
        if not safe_filename.endswith('.txt'):
            safe_filename += '.txt'
        
        filepath = os.path.join(self.experiment_path, safe_filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  📄 Saved text file: {safe_filename}")
        return filepath

    def list_artifacts(self):
        """
        List all artifacts saved in the current experiment.
        
        Returns:
            list: List of artifact filenames
        """
        if not os.path.exists(self.experiment_path):
            return []
        
        artifacts = []
        for file in os.listdir(self.experiment_path):
            filepath = os.path.join(self.experiment_path, file)
            if os.path.isfile(filepath):
                artifacts.append(file)
        
        return sorted(artifacts)

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