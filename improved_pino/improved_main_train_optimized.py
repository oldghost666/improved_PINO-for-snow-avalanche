
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
    print("\n=== PINO Avalanche Simulation Training Mode Selection ===")
    print("1. Data Loss Only Training (data_only)")
    print("   - Train using data loss only")
    print("   - Suitable for early stage training and base model establishment")
    print("")
    print("2. Physics-Constrained Training (physics_constrained)")
    print("   - Combine data loss and physics constraints")
    print("   - Includes physical laws like continuity and momentum conservation")
    print("   - Recommended for final training")
    print("")
    print("3. Progressive Training (progressive)")
    print("   - Start with data loss training, then add physics constraints")
    print("   - Balance training stability and physical consistency")
    print("   - Suitable for most scenarios")
    print("")
    while True:
        try:
            choice = input("Please select training mode (1-3): ").strip()
            if choice == '1':
                return 'data_only'
            elif choice == '2':
                return 'physics_constrained'
            elif choice == '3':
                return 'progressive'
            else:
                print("❌ Invalid choice, please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n\n❌ User cancelled operation")
            exit(0)
def get_training_parameters():
    print("\n=== Training Parameter Settings ===")
    while True:
        try:
            total_epochs = input(f"Please enter total training epochs (default: 200): ").strip()
            if total_epochs == "":
                total_epochs = 200
                break
            total_epochs = int(total_epochs)
            if total_epochs > 0:
                break
            else:
                print("❌ Total epochs must be greater than 0")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n❌ User cancelled operation")
            exit(0)
    while True:
        try:
            physics_start_epoch = input(f"Please enter physics constraint start epoch (default: 40): ").strip()
            if physics_start_epoch == "":
                physics_start_epoch = 40
                break
            physics_start_epoch = int(physics_start_epoch)
            if 0 <= physics_start_epoch <= total_epochs:
                break
            else:
                print(f"❌ Physics constraint start epoch must be between 0 and {total_epochs}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n❌ User cancelled operation")
            exit(0)
    print(f"\n✓ Training parameter settings completed:")
    print(f"  - Total training epochs: {total_epochs}")
    print(f"  - Physics constraint start epoch: {physics_start_epoch}")
    return total_epochs, physics_start_epoch
def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='PINO Avalanche Simulation Training Script')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--mode', type=str, choices=['data_driven', 'physics_constrained'], help='Training mode')
    parser.add_argument('--epochs', type=int, help='Training epochs')
    parser.add_argument('--physics-start', type=int, help='Physics constraint start epoch')
    parser.add_argument('--non-interactive', action='store_true', help='Non-interactive run')
    args = parser.parse_args()
    if args.config:
        config_path = args.config
    else:
        config_path = os.environ.get('PINO_CONFIG_PATH')
        if not config_path:
            base_dir = Path(__file__).parent
            config_path = str(base_dir / "configs" / "config_optimized.yaml")
            print(f"⚠️  No configuration file specified, using default path: {config_path}")
            print(f"   It is recommended to use the optimized configuration file for best training results")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if args.mode:
        selected_mode = args.mode
        print(f"✓ Using command-line specified training mode: {selected_mode}")
    elif args.non_interactive:
        selected_mode = 'physics_constrained'
        print(f"✓ Non-interactive mode, using default training mode: {selected_mode}")
    else:
        selected_mode = select_training_mode()
    config['training']['mode'] = selected_mode
    if args.epochs and args.physics_start is not None:
        total_epochs = args.epochs
        physics_start_epoch = args.physics_start
        print(f"✓ Using command-line specified training parameters: total epochs={total_epochs}, physics constraint start epoch={physics_start_epoch}")
    elif args.non_interactive:
        total_epochs = 200
        physics_start_epoch = 40
        print(f"✓ Non-interactive mode, using default training parameters: total epochs={total_epochs}, physics constraint start epoch={physics_start_epoch}")
    else:
        total_epochs, physics_start_epoch = get_training_parameters()
    config['training']['epochs'] = total_epochs
    config['training']['physics_constrained']['data_only_epochs'] = physics_start_epoch
    original_patience = config['training']['patience']
    print(f"\n⚠️  Important Reminder: Early stopping patience value read from configuration file: {original_patience}")
    print(f"   To modify, please edit the configuration file configs/config_optimized.yaml directly")
    print(f"\n📋 Configuration update completed:")
    print(f"  - Training mode: {selected_mode}")
    print(f"  - Total training epochs: {total_epochs}")
    print(f"  - Physics constraint start epoch: {physics_start_epoch}")
    cuda_available = torch.cuda.is_available()
    use_cuda = config.get('device', {}).get('use_cuda', True) and cuda_available
    force_cpu = config.get('device', {}).get('force_cpu', False)
    if force_cpu:
        device = torch.device('cpu')
        print(f"Using device: {device} (Forced to use CPU)")
    elif use_cuda:
        cuda_device = config.get('device', {}).get('cuda_device', 0)
        device = torch.device(f"cuda:{cuda_device}")
        print(f"Using device: {device} (CUDA available)")
    else:
        device = torch.device('cpu')
        if not cuda_available:
            print(f"Using device: {device} (CUDA unavailable, falling back to CPU)")
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