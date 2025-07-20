# Battery SoC Estimation Setup Guide

This guide will help you set up the battery state of charge estimation project for development and experimentation.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- At least 8GB RAM (16GB recommended)
- 5GB free disk space

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/danoushf/soc-estimation-fpga.git
cd soc-estimation-fpga
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. GPU Setup (Optional but Recommended)

If you have an NVIDIA GPU:

```bash
# Verify GPU is available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 5. Data Setup

1. Download the UniboPowertools dataset from here: https://data.mendeley.com/datasets/n6xg5fzsbv/1
2. Place the dataset in the appropriate directory structure:
   ```
   data/
   ├── battery-data/
   │   ├── 000-DM-3.0-4019-S/
   │   ├── 001-DM-3.0-4019-S/
   │   └── ...
   ```

### 6. Run the Notebook

```bash
# Launch Jupyter Lab
jupyter lab

# Or launch Jupyter Notebook
jupyter notebook
```

Open `soc_estimation_notebook.ipynb` and run the cells.

## Configuration

### Model Selection

In the notebook, you can choose between different architectures:

```python
# Choose model architecture
build_model = build_lstm_model  # LSTM
# build_model = build_bi_lstm_model  # Bidirectional LSTM
# build_model = build_1d_cnn_model  # 1D CNN
```

### Hyperparameter Tuning

Adjust the tuning parameters:

```python
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=20,  # Adjust number of trials
    executions_per_trial=1,
    directory='bayesian_optimization',
    project_name=experiment_name,
    overwrite=False
)
```

### Training Parameters

Modify training configuration:

```python
# Training parameters
window_size = 30  # Time steps in sliding window
stride = 1        # Window stride
batch_size = 2048 # Batch size for training
epochs = 100      # Maximum epochs
```

## Expected Results

After training, you should see:

1. **Model files**: `best_soc_model_*.h5`
2. **Evaluation results**: `*_evaluation_results.txt`
3. **Hyperparameters**: `best_hyperparameters_*.txt`
4. **Visualizations**: Interactive plots showing predicted vs actual SoC

## Troubleshooting

### Common Issues

1. **GPU Memory Error**:
   - Reduce batch size
   - Reduce model complexity
   - Use mixed precision training

2. **Out of Memory**:
   - Reduce window size
   - Use data generators
   - Increase system RAM

3. **Import Errors**:
   - Verify all dependencies are installed
   - Check Python version compatibility
   - Ensure virtual environment is activated

### Performance Tips

1. **For better performance**:
   - Use GPU acceleration
   - Increase batch size (if memory allows)
   - Use mixed precision training

2. **For faster experimentation**:
   - Reduce max_trials in hyperparameter tuning
   - Use smaller window sizes
   - Reduce the number of epochs

## Contact

For issues or questions, please open an issue on the GitHub repository.
