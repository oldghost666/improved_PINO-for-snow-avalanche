# Improved PINO Avalanche Dynamics Simulation Project

## Project Introduction
This project is a deep learning framework based on **Physics-Informed Neural Operator (PINO)**, specifically designed to simulate and predict avalanche dynamics. By combining the powerful feature extraction capabilities of the Fourier Neural Operator (FNO) with the physical constraints of the Shallow Water Equations, this project achieves high-precision and physically consistent simulation of avalanche height and velocity fields.

### Core Features
- **Advanced Architecture**: Adopts an improved FNO structure, introducing multi-scale feature fusion, channel attention mechanisms, and residual connections to significantly enhance the model's expressive power.
- **Physics-Driven**: Introduces a dimensionless physics loss function, directly imposing continuity and momentum conservation constraints in both frequency and spatial domains.
- **Progressive Training**: Supports a progressive training strategy from pure data-driven to physics-constrained, balancing convergence speed and physical realism.
- **Stable and Reliable**: Built-in numerical stability protection mechanisms, including output range clipping, gradient clipping, and mixed normalization strategies.
- **Industrial-Grade Configuration**: Comprehensive YAML configuration file management, supporting highly flexible hyperparameter adjustment.

---

## Project Structure
```text
improved_pino/
├── global_data_config.py           # Global data configuration and normalization management
├── improved_dataset.py             # Efficient data loading and preprocessing module
├── improved_model.py               # Implementation of the improved PINO network architecture
├── improved_physics_dimensionless.py # Dimensionless physics loss calculation module
├── improved_trainer.py             # Progressive training logic and multi-loss function management
├── improved_main_train_optimized.py # Main training entry script (supports interactive configuration)
├── config_optimized.yaml           # Optimized hyperparameter configuration file
└── validation/                     # Validation results and visualization output directory
```

---

## Core Workflow

### 1. Data Preparation and Normalization
The project manages data flow through `global_data_config.py`. Height, velocity components, and gradients use **Standard Normalization (Z-score)**, while DEM and physical parameters use **MinMax Normalization**. This hybrid strategy ensures a balanced sensitivity of the model to physical quantities of different scales.

### 2. Model Architecture
The PINO model implemented in `improved_model.py` includes:
- **SpectralConv2d**: The core Fourier convolution layer, capturing global correlations in the frequency domain.
- **ImprovedFNOBlock**: An enhanced operator block that captures local details through multi-scale convolutions and filters key physical features using channel attention.

### 3. Physical Constraints (Core of PINO)
`improved_physics_dimensionless.py` transforms the Shallow Water Equations into dimensionless loss terms.
- **Continuity Equation**: Ensures mass conservation.
- **Momentum Equation**: Ensures that force and motion status comply with fluid dynamics laws.
- **Dimensionless Processing**: Eliminates dimensional inconsistency caused by spatial scales (dx, dy) and time scales (dt), greatly improving training stability.

### 4. Training Strategy
The `ProgressiveTrainer` is used for a three-stage training process:
1. **Pure Data Stage**: Quickly fits the observed data distribution.
2. **Physics Introduction Stage**: Gradually increases the weight of physics loss to correct predictions that do not comply with physical laws.
3. **Fine-Tuning Stage**: Combines high-weight physics constraints with low learning rates for final optimization.

---

## Quick Start

### Environment Requirements
- Python 3.8+
- PyTorch 2.0+ (CUDA support recommended)
- h5py, numpy, pyyaml, matplotlib

### Running Training
You can start training via the interactive command line:
```bash
python improved_main_train_optimized.py
```
Alternatively, run non-interactively using a specified configuration:
```bash
python improved_main_train_optimized.py --config config_optimized.yaml --non-interactive
```

---

## Validation and Results
The project generates detailed validation reports in the `validation/` directory, including:
- **Dynamic Animations**: GIF animations of the avalanche evolution process.
- **Comparison Charts**: Cross-sectional comparisons between model predictions and ground truth.
- **Metric Analysis**: Comprehensive reports including MSE, MAE, and physics loss evolution curves.

---

## License
This project is licensed under the MIT Open Source License.
