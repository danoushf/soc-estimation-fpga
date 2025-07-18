# Trained Models

This directory contains the trained models from various experiments.

## Model Files

- `best_soc_model_*.h5`: Best performing models from hyperparameter optimization
- Each model file corresponds to a specific architecture and configuration

## Usage

To load a trained model:

```python
from tensorflow.keras.models import load_model

# Load the best model
model = load_model('best_soc_model_lstm_soc_estimation.h5')

# Make predictions
predictions = model.predict(test_data)
```

## Model Performance

Check the corresponding evaluation results in the `results/` directory for performance metrics of each model.
