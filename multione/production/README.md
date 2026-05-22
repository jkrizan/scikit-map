# Production Module

This module contains the production pipeline for the multione project, implementing a comprehensive system for data processing and monitoring.

## Overview

The production module provides a complete workflow for processing geospatial data with **Closed-form Continuous-time (CfC)** capabilities, monitoring, and logging.

## Core Components

### Main Files

- **`settings.py`** - Configuration management
- **`cfc_cell.py`** - CFC cell-level processing logic
- **`cfc.py`** - Continuous Feedback Control implementation
- **`data_utils.py`** - Data processing utilities and image handling
- **`production_utils.py`** - General production helper functions
- **`production_monitoring.py`** - System monitoring and metrics collection
- **`production_logger.py`** - Centralized logging functionality
- **`production.py`** - Main production pipeline orchestrator

## Usage

1. Configure settings in `settings.py`
2. Run production:

```bash
python production.py
```
## Output Structure

Generated outputs are stored in:
- `/mnt/nibble/gen_cog/arcov2/production/` - Main output directory for processed data

## Monitoring

The system includes built-in monitoring capabilities:
- Real-time processing metrics
- Error tracking and logging
- Performance monitoring

```bash
python production_monitoring.py
```

## Requirements

Ensure all dependencies are installed and the mount point structure is properly configured before running the production pipeline.