# Data Processing Module

This directory contains the data processing utilities for the battery SoC estimation project.

## Required Files

The following files should be placed in this directory for the project to work:

1. `unibo_powertools_data.py` - Contains the `UniboPowertoolsData` class
2. `model_data_handler.py` - Contains the `ModelDataHandler` class

## Purpose

- `UniboPowertoolsData`: Handles loading and preprocessing of the UniboPowertools dataset
- `ModelDataHandler`: Manages model input/output data preparation and transformations
- `CycleCols`: Enumeration for cycle column definitions

## Usage

```python
from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols
from data_processing.model_data_handler import ModelDataHandler

# Initialize data handler
dataset = UniboPowertoolsData(
    test_types=['S'],
    chunk_size=1000000,
    lines=[37, 40],
    charge_line=37,
    discharge_line=40,
    base_path=data_path
)

# Initialize model data handler
mdh = ModelDataHandler(dataset, [
    CycleCols.VOLTAGE,
    CycleCols.CURRENT,
    CycleCols.TEMPERATURE
])
```

## Note

These files are specific to the UniboPowertools dataset and may need to be adapted for other battery datasets. For more information, please visit this page:
https://data.mendeley.com/datasets/n6xg5fzsbv/1
