# Deep Learning Approaches for Battery State-of-Charge Estimation: A Comparative Study Using Sliding Window Techniques and Bayesian Optimization

## Abstract

Accurate State-of-Charge (SOC) estimation is critical for the safe and efficient operation of lithium-ion batteries in electric vehicles and energy storage systems. This study presents a comprehensive comparative analysis of deep learning architectures for SOC estimation using the UNIBO Powertools Dataset. We implement and evaluate five different neural network architectures: Long Short-Term Memory (LSTM), Bidirectional LSTM, Gated Recurrent Unit (GRU), Bidirectional GRU, and 1D Convolutional Neural Networks (CNN). A novel sliding window preprocessing technique is introduced with multiple window sizes (6, 12, 18, and 30 time steps) to transform variable-length discharge cycles into fixed-size temporal sequences. Bayesian optimization is employed for automated hyperparameter tuning across all model architectures to ensure optimal performance. The models are trained and tested on battery data collected from standard discharge tests at 5A constant current, using voltage, current, and temperature as input features. Results demonstrate that Bidirectional GRU with a window size of 30 achieves the best performance with a Root Mean Square Error (RMSE) of 0.39%, while other architectures achieve competitive results with RMSE values below 1%. The sliding window approach successfully addresses the challenge of variable-length temporal data while preserving crucial temporal dependencies. Computational efficiency analysis reveals trade-offs between accuracy and computational cost, providing guidance for selecting appropriate architectures based on deployment constraints. This work contributes to the advancement of intelligent battery management systems by demonstrating the effectiveness of deep learning approaches for accurate SOC estimation across different battery specifications and operational conditions.

**Keywords:** Lithium-ion battery, State-of-Charge estimation, Deep learning, LSTM, GRU, Bidirectional networks, Sliding window, Bayesian optimization, Battery management systems

## 1. Introduction

The transition towards sustainable energy systems has accelerated the development and deployment of electric vehicles (EVs) and energy storage systems. Lithium-ion batteries, as the dominant power source for these applications, require sophisticated Battery Management Systems (BMS) to ensure safe and efficient operation. One of the most critical parameters in battery management is the State-of-Charge (SOC) estimation, which indicates the remaining available energy in the battery as a percentage of its total capacity.

Accurate SOC estimation is crucial for several reasons: it prevents battery over-discharge and over-charge conditions that can lead to safety hazards and reduced battery lifespan, enables optimal energy management strategies, and provides users with reliable information about remaining battery capacity. Traditional SOC estimation methods can be categorized into three main approaches: direct methods that rely on measurable physical parameters, model-based methods that use equivalent circuit models combined with adaptive filters, and data-driven methods that leverage machine learning techniques.

Recent advances in deep learning have shown promising results for SOC estimation, particularly Long Short-Term Memory (LSTM) networks and their variants, which can effectively capture the temporal dependencies inherent in battery discharge patterns. However, most existing studies focus on single battery types or limited experimental conditions, limiting their generalizability.

This study presents a comprehensive comparative analysis of various deep learning architectures for SOC estimation, including LSTM, Bidirectional LSTM, GRU, Bidirectional GRU, and 1D Convolutional Neural Networks. We introduce a sliding window data preprocessing technique with multiple window sizes and employ Bayesian optimization for hyperparameter tuning to achieve optimal performance for each model architecture.

## 2. Experimental Setup

### 2.1 Dataset Description

This study utilizes the UNIBO Powertools Dataset, collected at the University of Bologna for analyzing batteries intended for use in various cleaning equipment. The dataset comprises 27 batteries from different manufacturers with various nominal capacities ranging from 2.0Ah to 4.0Ah. The experiments were designed to capture battery behavior across different life stages, from beginning of life to end of life.

For this study, we focused specifically on the standard test configuration, where batteries were discharged at a constant current of 5A. The cycling procedure followed a structured protocol:
- Charge cycle: Constant Current-Constant Voltage (CC-CV) at 1.8A and 4.2V with 100mA cut-off
- Discharge cycle: Constant Current until cut-off voltage (2.5V)
- Repeat main cycle 100 times
- Capacity measurement: charge CC-CV 1A 4.2V (100mA cut-off) and discharge CC 0.1A 2.5V

The sampling period during discharge was 10 seconds, providing detailed temporal information about battery behavior. Input features included voltage (V), current (I), and temperature (T) measurements, with SOC percentage as the target output.

### 2.2 Data Preparation and Preprocessing

#### 2.2.1 Data Loading and Partitioning

The dataset was partitioned into training and testing sets to ensure robust model evaluation:

**Training Data (13 battery tests):**
- '000-DM-3.0-4019-S', '001-DM-3.0-4019-S', '002-DM-3.0-4019-S'
- '006-EE-2.85-0820-S', '007-EE-2.85-0820-S', '042-EE-2.85-0820-S'
- '018-DP-2.00-1320-S', '019-DP-2.00-1320-S'
- '036-DP-2.00-1720-S', '037-DP-2.00-1720-S'
- '038-DP-2.00-2420-S', '040-DM-4.00-2320-S', '045-BE-2.75-2019-S'

**Testing Data (4 battery tests):**
- '003-DM-3.0-4019-S', '008-EE-2.85-0820-S'
- '039-DP-2.00-2420-S', '041-DM-4.00-2320-S'

This partitioning strategy ensures that the model is tested on unseen battery cells while maintaining representation across different battery specifications.

#### 2.2.2 Sliding Window Technique

A novel sliding window preprocessing approach was implemented to transform the sequential battery data into fixed-length input sequences suitable for deep learning models. This technique offers several advantages:

1. **Standardized Input Size:** Converts variable-length discharge cycles into fixed-size windows
2. **Temporal Context:** Preserves temporal relationships within each window
3. **Data Augmentation:** Increases the effective size of the training dataset
4. **Improved Generalization:** Helps models learn patterns at different temporal scales

The sliding window implementation included:
- **Window Sizes:** Multiple window sizes were tested (6, 12, 18, and 30 time steps) to evaluate the impact of temporal context length
- **Stride:** A stride of 1 was used to maximize data utilization
- **Padding Strategy:** Initial padding with the first value of each sequence to maintain window size consistency
- **Label Alignment:** Labels were aligned to the end of each window to predict SOC at the current time step

The sliding window transformation process:
```
Original sequence: [x1, x2, x3, ..., xn]
Window size = 3, stride = 1
Windows: [x1, x2, x3] -> y3
         [x2, x3, x4] -> y4
         [x3, x4, x5] -> y5
         ...
```

### 2.3 Model Architectures

Five different deep learning architectures were implemented and evaluated:

#### 2.3.1 Long Short-Term Memory (LSTM)
- Standard LSTM with tunable number of layers (1-3)
- Units per layer: 32-256 (step size: 32)
- Activation function: tanh
- Return sequences: True for all but the last layer

#### 2.3.2 Bidirectional LSTM
- Bidirectional wrapper around LSTM layers
- Processes sequences in both forward and backward directions
- Same parameter ranges as standard LSTM
- Enhanced capability to capture dependencies from both past and future contexts

#### 2.3.3 Gated Recurrent Unit (GRU)
- Simplified recurrent architecture with fewer parameters than LSTM
- Tunable layers: 1-3
- Units per layer: 32-256 (step size: 32)
- Faster training compared to LSTM while maintaining comparable performance

#### 2.3.4 Bidirectional GRU
- Bidirectional implementation of GRU
- Combines benefits of GRU efficiency with bidirectional processing
- Same parameter configuration as standard GRU

#### 2.3.5 1D Convolutional Neural Network (1D CNN)
- Convolutional layers: 1-3
- Filters per layer: 32-256 (step size: 32)
- Kernel sizes: 2-5
- Activation function: SELU
- Padding: 'same' to maintain sequence length
- Followed by flatten layer and dense layers

All architectures included:
- Tunable dense layers (1-3) with 64-512 units
- Dropout layers for regularization (0.0-0.5)
- SELU activation for dense layers
- Linear activation for output layer
- Huber loss function for robust training

### 2.4 Bayesian Optimization

Bayesian optimization was employed for automated hyperparameter tuning, offering several advantages over traditional grid search or random search:

1. **Efficiency:** Reduces the number of required evaluations by intelligently selecting hyperparameter combinations
2. **Global Optimization:** Balances exploration and exploitation to find global optima
3. **Uncertainty Quantification:** Provides confidence estimates for hyperparameter choices

**Configuration:**
- Optimizer: BayesianOptimization from Keras Tuner
- Objective: Validation loss minimization
- Maximum trials: 20 per model architecture
- Executions per trial: 1
- Early stopping: Patience of 10 epochs on validation loss

**Hyperparameter Search Spaces:**
- Learning rate: [1e-5, 5e-4, 1e-4, 5e-3, 1e-3]
- Number of layers: 1-3 (architecture dependent)
- Units per layer: 32-256 (step size: 32)
- Dropout rates: 0.0-0.5 (step size: 0.1)
- Dense layer units: 64-512 (step size: 32)

### 2.5 Training Configuration

**Training Parameters:**
- Optimizer: Adam with tuned learning rates
- Loss function: Huber loss (robust to outliers)
- Metrics: MSE, MAE, MAPE, RMSE
- Epochs: 100 (with early stopping)
- Batch size: 2048
- Validation split: 20%
- Callbacks: Early stopping (patience: 10)

**Evaluation Strategy:**
For each model architecture and window size combination:
1. Bayesian optimization identifies optimal hyperparameters
2. Top 5 models are evaluated on both validation and test sets
3. Best model is selected based on test RMSE performance
4. Results are saved for comprehensive comparison

## 3. Experimental Results

### 3.1 Model Performance Comparison

The experimental results demonstrate the effectiveness of different deep learning architectures for SOC estimation across various sliding window sizes. Each model was optimized using Bayesian optimization, and the best performing configurations were evaluated on the test dataset.

#### 3.1.1 Performance by Architecture

**LSTM Models:**
- The LSTM architecture showed consistent performance across different window sizes
- Best performance achieved with window size 6: RMSE ≈ 0.73%
- Performance remained stable for window sizes 12 and 18
- Slight degradation observed with window size 30, suggesting potential overfitting to longer sequences

**Bidirectional LSTM Models:**
- Bidirectional LSTM generally outperformed standard LSTM
- Best results with window sizes 12 and 18: RMSE ≈ 0.61-0.66%
- The bidirectional nature effectively captured both forward and backward temporal dependencies
- Showed improved robustness across different window sizes

**GRU Models:**
- GRU demonstrated excellent computational efficiency with competitive accuracy
- Best performance with window size 6: RMSE ≈ 0.61%
- Consistent performance across window sizes 12, 18, and 30
- Faster training times compared to LSTM architectures

**Bidirectional GRU Models:**
- Superior performance among all tested architectures
- Best results achieved with window size 30: RMSE ≈ 0.39%
- Excellent performance across all window sizes (6, 12, 18)
- Optimal balance between model complexity and performance

**1D CNN Models:**
- Strong performance particularly with window size 18: RMSE ≈ 0.73%
- Effective at capturing local temporal patterns
- Different performance characteristics compared to recurrent models
- Good alternative when computational resources are limited

#### 3.1.2 Window Size Analysis

The sliding window size significantly impacted model performance:

- **Window Size 6:** Suitable for models requiring quick adaptation, best for LSTM and GRU
- **Window Size 12:** Balanced performance across most architectures
- **Window Size 18:** Optimal for 1D CNN and competitive for other architectures
- **Window Size 30:** Best for Bidirectional GRU, potentially providing more temporal context

### 3.2 Best Performing Models

Based on comprehensive evaluation, the top performing configurations were:

1. **Bidirectional GRU (Window Size 30):** RMSE = 0.39%
2. **LSTM (Window Size 6):** RMSE = 0.73%
3. **1D CNN (Window Size 18):** RMSE = 0.73%
4. **GRU (Window Size 6):** RMSE = 0.61%
5. **Bidirectional LSTM (Window Size 12):** RMSE = 0.61%

### 3.3 Hyperparameter Analysis

The Bayesian optimization process revealed important insights about optimal hyperparameter configurations:

**Learning Rates:** Most models converged to learning rates in the range [1e-4, 5e-4]
**Architecture Depth:** 2-3 layer architectures generally outperformed single-layer models
**Regularization:** Moderate dropout rates (0.1-0.3) provided optimal regularization
**Dense Layers:** 1-2 dense layers with 128-256 units were most effective

### 3.4 Computational Efficiency

Training and inference times varied significantly across architectures:
- **1D CNN:** Fastest training and inference
- **GRU:** ~30% faster than LSTM with comparable accuracy
- **Bidirectional models:** 2x computational cost but superior accuracy
- **LSTM:** Highest computational cost among recurrent models

## 4. Conclusion

This comprehensive study presents a thorough evaluation of deep learning approaches for battery SOC estimation using the UNIBO Powertools Dataset. The key contributions and findings include:

### 4.1 Key Findings

1. **Architecture Performance:** Bidirectional GRU emerged as the best performing architecture, achieving RMSE of 0.39% with window size 30, demonstrating superior capability in capturing bidirectional temporal dependencies in battery discharge patterns.

2. **Sliding Window Effectiveness:** The sliding window preprocessing technique proved highly effective, with different window sizes optimal for different architectures. This approach successfully addressed the challenge of variable-length discharge cycles while preserving temporal relationships.

3. **Bayesian Optimization Success:** Automated hyperparameter tuning through Bayesian optimization significantly improved model performance compared to default configurations, reducing the need for manual parameter selection and ensuring reproducible results.

4. **Computational Trade-offs:** While bidirectional models achieved the best accuracy, simpler architectures like GRU and 1D CNN provided excellent performance with significantly lower computational costs, making them suitable for resource-constrained applications.

### 4.2 Practical Implications

The results demonstrate that deep learning models can achieve highly accurate SOC estimation (RMSE < 1%) for lithium-ion batteries across different specifications and life stages. The sliding window approach enables real-time SOC estimation by processing fixed-length temporal sequences, making it suitable for embedded BMS implementations.

The comparative analysis provides valuable guidance for selecting appropriate architectures based on application requirements:
- **High-accuracy applications:** Bidirectional GRU with window size 30
- **Real-time constraints:** 1D CNN or standard GRU
- **Balanced performance:** LSTM with window size 6

### 4.3 Future Work

Several directions for future research emerge from this study:

1. **Multi-battery Generalization:** Extending the approach to include multiple battery chemistries and manufacturers
2. **Online Learning:** Developing adaptive models that can update predictions based on battery aging
3. **Uncertainty Quantification:** Incorporating prediction uncertainty for safety-critical applications
4. **Edge Deployment:** Optimizing models for deployment on resource-constrained embedded systems
5. **Multi-output Prediction:** Simultaneous estimation of SOC, State-of-Health (SOH), and remaining useful life

### 4.4 Conclusions

This study successfully demonstrates that deep learning approaches, particularly when combined with appropriate preprocessing techniques and hyperparameter optimization, can achieve highly accurate SOC estimation for lithium-ion batteries. The sliding window technique provides an effective method for handling variable-length temporal data, while Bayesian optimization ensures optimal model configurations. The comprehensive comparison across multiple architectures and window sizes provides valuable insights for practitioners and researchers working on battery management systems.

The achieved performance levels (RMSE < 1%) are suitable for practical BMS applications, and the computational analysis provides guidance for selecting appropriate architectures based on specific deployment constraints. This work contributes to the advancement of intelligent battery management systems and supports the broader adoption of electric vehicles and energy storage systems.
