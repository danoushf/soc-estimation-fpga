# Battery State of Charge (SoC) Estimation for FPGA Implementation

**Lightweight Deep Learning-Based SOC Estimation for Lithium-Ion Batteries on ZCU104 FPGA**

This repository contains a comprehensive implementation of battery State of Charge (SoC) estimation using various deep learning models including LSTM, Bidirectional LSTM, GRU, Bidirectional GRU, and 1D CNN. The project is designed with FPGA implementation in mind and includes hyperparameter optimization using Bayesian optimization.

## 📋 Overview

Battery State of Charge estimation is crucial for battery management systems, especially in electric vehicles and energy storage applications. This project implements and compares multiple deep learning architectures to accurately predict battery SoC using voltage, current, and temperature data.

## 🔧 Features

- **Multiple Model Architectures**: 
  - LSTM (Long Short-Term Memory)
  - Bidirectional LSTM
  - GRU (Gated Recurrent Unit)
  - Bidirectional GRU
  - 1D CNN (Convolutional Neural Network)

- **Advanced Preprocessing**:
  - Sliding window approach for time series data
  - Data padding and alignment
  - Configurable window sizes and stride

- **Hyperparameter Optimization**:
  - Bayesian optimization using Keras Tuner
  - Automated model selection
  - Performance evaluation across multiple metrics

- **Comprehensive Evaluation**:
  - Multiple metrics: MSE, MAE, MAPE, RMSE
  - Train/validation/test split
  - Model comparison and selection

## 🛠 Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: At least 5GB free space for models and data

## 📊 Dataset

The project uses the UniboPowertools dataset with the following characteristics:

- **Input Features**: Voltage, Current, Temperature
- **Output**: State of Charge (SoC) percentage
- **Test Types**: 'S' (Specific discharge cycles)
- **Battery Lines**: 37 (charge), 40 (discharge)

### Data Split

**Training Data:**
- 000-DM-3.0-4019-S, 001-DM-3.0-4019-S, 002-DM-3.0-4019-S
- 006-EE-2.85-0820-S, 007-EE-2.85-0820-S
- 018-DP-2.00-1320-S, 019-DP-2.00-1320-S
- 036-DP-2.00-1720-S, 037-DP-2.00-1720-S
- 038-DP-2.00-2420-S, 040-DM-4.00-2320-S
- 042-EE-2.85-0820-S, 045-BE-2.75-2019-S

**Testing Data:**
- 003-DM-3.0-4019-S, 008-EE-2.85-0820-S
- 039-DP-2.00-2420-S, 041-DM-4.00-2320-S

## 🚀 Usage

### Running the Notebook

1. **Setup Environment**:
   ```bash
   # Configure Python environment
   # Install required packages
   pip install -r requirements.txt
   ```

2. **Configure Data Path**:
   ```python
   # Update data_path in the notebook
   data_path = "../"  # or your specific path
   ```

3. **Run the Notebook**:
   - Open `soc_estimation_notebook.ipynb` in Jupyter Lab or VS Code
   - Execute cells sequentially
   - Monitor training progress and results

### Model Configuration

The notebook includes several configurable parameters:

```python
# Sliding window parameters
window_size = 30  # Time steps in each window
stride = 1        # Step size for window sliding

# Model selection (choose one)
EXPERIMENT = "bi_lstm_SW 18"     # Bidirectional LSTM
EXPERIMENT = "bi_gru sw 18"      # Bidirectional GRU
EXPERIMENT = "GRU SW 18"         # GRU
EXPERIMENT = "LSTM_ba_slidingwindows_12"  # LSTM
EXPERIMENT = "1D CNN SW 18"      # 1D CNN
```

### Hyperparameter Optimization

The project uses Bayesian optimization for automated hyperparameter tuning:

```python
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='bayesian_optimization',
    project_name='your_project_name'
)
```

## 📈 Model Architectures

### 1. LSTM Model
- Tunable number of LSTM layers (1-3)
- Configurable units per layer (32-256)
- Dropout regularization (0.0-0.5)
- Dense layers with SELU activation

### 2. Bidirectional LSTM
- Bidirectional processing for better context
- Similar architecture to LSTM but with bidirectional layers
- Enhanced temporal feature extraction

### 3. GRU Model
- Lighter alternative to LSTM
- Fewer parameters, faster training
- Comparable performance to LSTM

### 4. Bidirectional GRU
- Combines GRU efficiency with bidirectional processing
- Optimal for sequences with temporal dependencies

### 5. 1D CNN
- Convolutional layers for feature extraction
- Flatten layer before dense layers
- Suitable for local pattern recognition

## 📊 Performance Metrics

The models are evaluated using:

- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **RMSE** (Root Mean Squared Error)

## 🔧 FPGA Implementation Considerations

This project is designed with FPGA implementation in mind:

1. **Model Quantization**: Use appropriate precision for FPGA deployment
2. **Memory Optimization**: Sliding window approach reduces memory requirements
3. **Pipeline Design**: Sequential processing suitable for FPGA pipelines
4. **Hardware Acceleration**: Models can be accelerated using FPGA DSP blocks

## 📁 File Structure

```
soc-estimation-fpga/
├── README.md
├── requirements.txt
├── soc_estimation_notebook.ipynb
├── data_processing/
│   ├── unibo_powertools_data.py
│   └── model_data_handler.py
├── models/
│   ├── best_soc_model_*.h5
│   └── evaluation_results/
├── bayesian_optimization/
└── results/
    ├── evaluation_results.txt
    └── best_hyperparameters.txt
```

## 🎯 Results

The project generates:

1. **Trained Models**: Saved as `.h5` files
2. **Evaluation Results**: Performance metrics for each model
3. **Hyperparameters**: Best configuration for each architecture
4. **Visualizations**: Training and testing predictions

## 🔍 Visualization

The notebook includes interactive visualizations using Plotly:

- Training vs. predicted SoC curves
- Testing performance visualization
- Model comparison plots

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the repository owner.

## 🙏 Acknowledgments

- UniboPowertools dataset contributors
- TensorFlow and Keras development teams
- Keras Tuner for hyperparameter optimization
- Plotly for interactive visualizations

## 📚 References

- https://github.com/KeiLongW/battery-state-estimation
- https://data.mendeley.com/datasets/n6xg5fzsbv/1
- [FPGA Implementation of Neural Networks]
- [Time Series Analysis with Deep Learning]
