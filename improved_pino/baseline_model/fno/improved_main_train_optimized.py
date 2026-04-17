
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import threading
from collections import defaultdict
from improved_dataset import ImprovedAvalancheDataset
from improved_model import ImprovedPINO
from improved_physics_dimensionless import DimensionlessPhysicsLoss, create_dimensionless_physics_loss
from global_data_config import GlobalDataConfig
from model_validation import ModelValidator
from improved_trainer import ProgressiveTrainer, OptimizedLossFunction
def select_training_mode():
    print("\n=== PINO Avalanche Training Mode Selection ===")
    print("1. Data-only Training (data_only)")
    print("   - Train using data loss only")
    print("   - Suitable for early stage training")
    print("")
    print("2. Physics-constrained Training (physics_constrained)")
    print("   - Combine data loss and physics constraints")
    print("   - Includes continuity and momentum laws")
    print("   - Recommended for final training")
    print("")
    print("3. Progressive Training (progressive)")
    print("   - Data loss first, then physics")
    print("   - Balance stability and consistency")
    print("   - Suitable for most scenarios")
    print("")
    while True:
        try:
            choice = input("Select training mode (1-3): ").strip()
            if choice == '1':
                return 'data_only'
            elif choice == '2':
                return 'physics_constrained'
            elif choice == '3':
                return 'progressive'
            else:
                print("[Error] Invalid choice, enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n\n[Error] User cancelled")
            exit(0)
def select_sampling_mode():
    print("\n=== Sampling mode selection ===")
    print("1. Single-step sampling training (Single-step)")
    print("   - Predict the next time step each time")
    print("   - Stable training, suitable for base models")
    print("")
    print("2. Multi-step autoregressive training (Multi-step)")
    print("   - Autoregressive prediction of multiple time steps")
    print("   - Better long-term evolution modeling")
    print("   - Requires Scheduled Sampling support")
    print("")
    while True:
        try:
            choice = input("Please select sampling mode (1-2): ").strip()
            if choice == '1':
                return False, 1  
            elif choice == '2':
                while True:
                    try:
                        steps = input("Please enter multi-step prediction steps (default: 5): ").strip()
                        if steps == "":
                            prediction_steps = 5
                            break
                        prediction_steps = int(steps)
                        if prediction_steps > 1:
                            break
                        else:
                            print("[Error] Multi-step prediction steps must be greater than 1")
                    except ValueError:
                        print("[Error] Enter a valid number")
                print(f"[Check] Multi-step prediction steps set to: {prediction_steps}")
                return True, prediction_steps
            else:
                print("[Error] Invalid choice, please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n\n[Error] User cancelled")
            exit(0)
def get_training_parameters():
    print("\n=== Training Parameters ===")
    while True:
        try:
            total_epochs = input(f"Enter total epochs (default: 200): ").strip()
            if total_epochs == "":
                total_epochs = 200
                break
            total_epochs = int(total_epochs)
            if total_epochs > 0:
                break
            else:
                print("[Error] Total epochs must be > 0")
        except ValueError:
            print("[Error] Enter a valid number")
        except KeyboardInterrupt:
            print("\n\n[Error] User cancelled")
            exit(0)
    while True:
        try:
            physics_start_epoch = input(f"Enter physics start epoch (default: 40): ").strip()
            if physics_start_epoch == "":
                physics_start_epoch = 40
                break
            physics_start_epoch = int(physics_start_epoch)
            if 0 <= physics_start_epoch <= total_epochs:
                break
            else:
                print(f"[Error] Physics start epoch must be between 0 and{total_epochs}")
        except ValueError:
            print("[Error] Enter a valid number")
        except KeyboardInterrupt:
            print("\n\n[Error] User cancelled")
            exit(0)
    print(f"\n[Check] Training parameters set:")
    print(f"  - Total training epochs: {total_epochs}")
    print(f"  - physics start epoch: {physics_start_epoch}")
    return total_epochs, physics_start_epoch
def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='PINO Avalanche Training Script')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--mode', type=str, choices=['data_driven', 'physics_constrained'], help='Training mode')
    parser.add_argument('--epochs', type=int, help='Training epochs')
    parser.add_argument('--physics-start', type=int, help='physics start epoch')
    parser.add_argument('--non-interactive', action='store_true', help='Non-interactive run')
    args = parser.parse_args()
    if args.config:
        config_path = args.config
    else:
        config_path = os.environ.get('PINO_CONFIG_PATH')
        if not config_path:
            base_dir = Path(__file__).parent
            config_path = str(base_dir / "configs" / "config_optimized.yaml")
            print(f"[Warning]  No config specified, using default: {config_path}")
            print(f"   Recommend using optimized config")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if args.mode:
        selected_mode = args.mode
        print(f"[Check] Using command-line training mode: {selected_mode}")
    elif args.non_interactive:
        selected_mode = 'physics_constrained'
        print(f"[Check] Non-interactive, using default mode: {selected_mode}")
    else:
        selected_mode = select_training_mode()
    config['training']['mode'] = selected_mode
    if args.non_interactive:
        use_multi_step = config.get('data', {}).get('multi_step_sampling', {}).get('enable', False)
        prediction_steps = config.get('data', {}).get('multi_step_sampling', {}).get('prediction_steps', 1)
        mode_name = "multi-step sampling" if use_multi_step else "single-step sampling"
        print(f"[Check] Non-interactive mode, reading sampling mode from config: {mode_name}")
        if use_multi_step:
            print(f"  Prediction steps: {prediction_steps}")
    else:
        use_multi_step, prediction_steps = select_sampling_mode()
    if 'data' not in config:
        config['data'] = {}
    if 'multi_step_sampling' not in config['data']:
        config['data']['multi_step_sampling'] = {}
    config['data']['multi_step_sampling']['enable'] = use_multi_step
    config['data']['multi_step_sampling']['prediction_steps'] = prediction_steps
    if use_multi_step:
        config['data']['multi_step_sampling']['mode'] = 'multi_step'
        config['data']['sequence_length'] = prediction_steps + 1  
    else:
        config['data']['multi_step_sampling']['mode'] = 'single_step'
        config['data']['sequence_length'] = 1
    print(f"[Check] Sampling configuration updated: Mode={config['data']['multi_step_sampling']['mode']}, "
          )
    if args.epochs and args.physics_start is not None:
        total_epochs = args.epochs
        physics_start_epoch = args.physics_start
        print(f"[Check] Using command-line parameters: total epochs={total_epochs}, physics start epoch={physics_start_epoch}")
    elif args.non_interactive:
        total_epochs = 200
        physics_start_epoch = 40
        print(f"[Check] Non-interactive, using default parameters: total epochs={total_epochs}, physics start epoch={physics_start_epoch}")
    else:
        total_epochs, physics_start_epoch = get_training_parameters()
    config['training']['epochs'] = total_epochs
    config['training']['physics_constrained']['data_only_epochs'] = physics_start_epoch
    original_patience = config['training']['patience']
    print(f"\n[Warning]  Early stopping patience read from config: {original_patience}")
    print(f"   Edit config file to modify")
    print(f"\n[Config] Configuration updated:")
    print(f"  - Training mode: {selected_mode}")
    print(f"  - Sampling mode: {'multi-step sampling' if use_multi_step else 'single-step sampling'}")
    if use_multi_step:
        print(f"  - Prediction steps: {prediction_steps}")
    print(f"  - Total training epochs: {total_epochs}")
    print(f"  - physics start epoch: {physics_start_epoch}")
    cuda_available = torch.cuda.is_available()
    use_cuda = config.get('device', {}).get('use_cuda', True) and cuda_available
    force_cpu = config.get('device', {}).get('force_cpu', False)
    if force_cpu:
        device = torch.device('cpu')
        print(f"Using device: {device} (Forced CPU)")
    elif use_cuda:
        cuda_device = config.get('device', {}).get('cuda_device', 0)
        device = torch.device(f"cuda:{cuda_device}")
        print(f"Using device: {device} (CUDA available)")
    else:
        device = torch.device('cpu')
        if not cuda_available:
            print(f"Using device: {device} (CUDA unavailable, fallback to CPU)")
        else:
            print(f"Using device: {device} (Configured to use CPU)")
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Set random seed: {seed}")
    print(f"Training mode: {config['training']['mode']}")
    print(f"Group by time pair: {config['data'].get('group_by_time_pair', True)}")
    print(f"Data directory: {config['paths']['data_dir']}")
    print(f"Checkpoint directory: {config['paths']['checkpoint_dir']}")
    print(f"Log directory: {config['paths']['log_dir']}")
    print(f"Results directory: {config['paths']['results_dir']}")
    trainer = ProgressiveTrainer(config, device)
    trainer.train()
if __name__ == "__main__":
    main()