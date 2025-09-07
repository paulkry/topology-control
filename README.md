# Topology Control
MIT SGI 2025 topology control project.

A modular, config-driven ML pipeline for 3D shape classification using PyTorch. Features auto-discovery of mesh files, configurable train/val splitting, and comprehensive artifact management.  Please see [part 1](https://summergeometry.org/sgi2025/topology-control-training-a-deepsdf-1-2/) and [part 2](https://summergeometry.org/sgi2025/topology-control-pathfinding-for-genus-preservation-2-2/) of our blog posts for details.  

## Features

- **Modular Pipeline**: Data processing → Model building → Training → Evaluation
- **Auto-Discovery**: Automatically finds and processes mesh files in raw data directory
- **Config-Driven**: Fully configurable via YAML files
- **Artifact Management**: Saves experiment results, model info, and training plots
- **3D Shape Processing**: Converts meshes to point clouds with signed distance fields
- **Flexible Architecture**: Supports configurable models

## Setup

### Python Environment

This project requires Python 3.10. You can set up the environment using either conda or venv:

#### Option 1: Conda (Recommended)
```bash
conda create --name topologycontrol python=3.10
conda activate topologycontrol
```

#### Option 2: Virtual Environment
```bash
python3.10 -m venv topologycontrol
source topologycontrol/bin/activate  # On Linux/Mac
# or
topologycontrol\Scripts\activate     # On Windows
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Required Packages

The `requirements.txt` includes:
- `torch` - PyTorch for deep learning
- `numpy` - Numerical computing
- `meshio` - Mesh file I/O
- `polyscope` - 3D visualization
- `matplotlib` - Plotting and visualization
- `pyyaml` - YAML configuration parsing
- `libigl` - Geometry processing (if available)

**Note**: Some packages like `triangle` may need to be installed separately if geometry processing fails.

## Project Structure

```
topology-control/
├── main.py                 # Main pipeline entry point
├── config/
│   └── config.yaml        # Configuration file
├── data/
│   ├── raw/              # Raw mesh files (.obj)
│   └── processed/        # Processed data (train/val splits)
├── src/
│   ├── CPipelineOrchestrator.py  # Main pipeline controller
│   ├── CDataProcessor.py         # Data processing and mesh handling
│   ├── CArchitectureManager.py   # Model architecture definitions
│   ├── CModelTrainer.py          # Training and validation logic
│   ├── CEvaluator.py            # Model evaluation
│   ├── CGeometryUtils.py        # 3D geometry utilities
│   └── CArtifactManager.py      # Experiment artifact management
└── artifacts/            # Generated experiment artifacts
```

## Usage

### Quick Start

1. Place your mesh files (`.obj` format) in `data/raw/`
2. Configure the pipeline in `config/config.yaml`
3. Run the pipeline:

```bash
python main.py
```

### Configuration

Edit `config/config.yaml` to customize:

- **Data paths**: Raw and processed data directories
- **Model settings**: Architecture, input/output dimensions
- **Training parameters**: Learning rate, batch size, epochs
- **Processing options**: Point cloud sampling, train/val split ratio
- **Pipeline control**: Skip specific steps for debugging

### Example Configuration

```yaml
# Basic setup
home: /path/to/topology-control

# Model configuration
model_config:
  skip_building: false
  model_name: mlp
  input_dim: 3000      # 1000 points × 3 coordinates
  hidden_dims: [512, 256, 128]
  output_dim: 1
  max_points: 1000     # Fixed number of points per shape

# Training parameters
trainer_config:
  skip_training: false
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 50
  optimizer: adam
  loss_function: mse
```

### Data Processing

The pipeline automatically:
1. Discovers all `.obj` files in `data/raw/`
2. Converts meshes to point clouds with signed distance fields
3. Splits data into train/validation sets
4. Saves processed data to `data/processed/train/` and `data/processed/val/`

### Artifacts

Each experiment generates timestamped artifacts in `artifacts/experiment_YYYYMMDD_HHMMSS/`:
- `pipeline_summary.txt` - Overall pipeline execution summary
- `model_architecture.txt` - Model structure and parameter counts
- `training_results.txt` - Training metrics and loss curves
- `error_report.txt` - Error details if pipeline fails

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed
2. **Tensor size mismatch**: Check `max_points` configuration matches model `input_dim`
3. **No mesh files found**: Verify `.obj` files are in `data/raw/` directory
4. **CUDA errors**: Set device explicitly or ensure GPU drivers are updated

### Environment Issues

If you encounter package conflicts:
```bash
# Clean conda environment
conda remove --name topologycontrol --all
conda create --name topologycontrol python=3.10
conda activate topologycontrol
pip install -r requirements.txt
```

### Data Issues

If processing fails:
- Ensure mesh files are valid `.obj` format
- Check file permissions in data directories
- Verify sufficient disk space for processed data

## Development

### Adding New Models

1. Implement model class in `CArchitectureManager.py`
2. Add model configuration to `config.yaml`
3. Update `get_model()` method to handle new architecture

### Extending Data Processing

1. Modify `CDataProcessor.py` for new data formats
2. Update `CGeometryUtils.py` for new geometry operations
3. Adjust dataset class in `CModelTrainer.py` if needed

## License

MIT License - See LICENSE file for details.

