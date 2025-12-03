
# ARCOv2 Multitemporal CFC Neural Network

This project implements a **Closed-form Continuous-time (CfC)** neural network for processing multitemporal satellite data from the ARCOv2 (Analysis Ready Cloud Optimized version 2) dataset.

## Overview

The system processes Landsat and MODIS satellite imagery to predict vegetation indices (like FPAR) using a neural network architecture that can handle irregular time series with missing data.

## Key Components

### Core Architecture
- **CfC Cell** (`cfc_cell.py`): Implements the Closed-form Continuous-time RNN cell
  - class `CfCCell`
- **CfC Model** (`cfc.py`): Main neural network model with forward/backward processing capabilities
  - class `CfcModel`
- **Dataset** (`cfc.py`): Custom PyTorch dataset for handling multitemporal satellite data
  - class `ArcoV2Dataset`

### Data Processing
- **Sampling** (`cfc_sample.py`): Extracts and samples data from satellite tiles stored in tiff's to Zarr format
  - `def sample_tiles`, `def sample_surface_water`
- **Utilities** (`utils.py`): Data loading, masking, temperature modeling, and preprocessing functions
- **Settings** (`settings.py`): Configuration parameters for data paths, processing options, and model hyperparameters

### Training
- **Distributed Training** (`cfc_train_ray.py`): Ray-based distributed training framework with GPU support

## Features

### Data Handling
- Processes Landsat ARD (Analysis Ready Data) and MODIS NDVI time series
- Handles missing data through quality masking and MODIS-based filtering
- Incorporates geometric temperature modeling based on latitude, elevation, and day-of-year
- Supports water masking using JRC Global Surface Water data

### Model Architecture
- CfC cells with configurable backbone networks
- Bidirectional processing (forward/backward through time)
- Mixed memory architecture combining CfC and LSTM cells
- Handles irregular time series with variable time spans

### Training Infrastructure
- Multi-GPU distributed training using Ray
- Memory-efficient data loading with caching
- Automatic checkpointing and resuming
- Configurable hyperparameter search

## Usage

### Data Preparation
1. Run `cfc_sample.py` to extract and sample data from satellite tiles
2. Optionally add surface water masking using `sample_surface_water()`

### Training
```python
# Configure training parameters in cfc_train_ray.py
python cfc_train_ray.py
```

### Key Configuration
- Input features: Vegetation indices + MODIS NDVI + geometric temperature
- Sequence length: 12 time steps
- Years: 2000-2024
- Tiles: Global Landsat ARD tiles

## Dependencies
- PyTorch
- Ray (distributed training)
- Zarr (data storage)
- Rasterio (geospatial data)
- NumPy, XArray, Pandas

