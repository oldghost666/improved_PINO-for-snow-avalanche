
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
    print("\n=== PINO雪崩模拟训练模式选择 ===")
    print("1. 仅数据损失训练 (data_only)")
    print("   - 仅使用数据损失进行训练")
    print("   - 适合初期训练和基础模型建立")
    print("")
    print("2. 物理约束训练 (physics_constrained)")
    print("   - 结合数据损失和物理约束")
    print("   - 包含连续性方程、动量守恒等物理定律")
    print("   - 推荐用于最终训练")
    print("")
    print("3. 渐进式训练 (progressive)")
    print("   - 先进行数据损失训练，后加入物理约束")
    print("   - 平衡训练稳定性和物理一致性")
    print("   - 适合大多数场景")
    print("")
    while True:
        try:
            choice = input("请选择训练模式 (1-3): ").strip()
            if choice == '1':
                return 'data_only'
            elif choice == '2':
                return 'physics_constrained'
            elif choice == '3':
                return 'progressive'
            else:
                print("❌ 无效选择，请输入1、2或3")
        except KeyboardInterrupt:
            print("\n\n❌ 用户取消操作")
            exit(0)
def get_training_parameters():
    print("\n=== 训练参数设置 ===")
    while True:
        try:
            total_epochs = input(f"请输入总训练轮数 (默认: 200): ").strip()
            if total_epochs == "":
                total_epochs = 200
                break
            total_epochs = int(total_epochs)
            if total_epochs > 0:
                break
            else:
                print("❌ 训练轮数必须大于0")
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n❌ 用户取消操作")
            exit(0)
    while True:
        try:
            physics_start_epoch = input(f"请输入物理约束开启轮数 (默认: 40): ").strip()
            if physics_start_epoch == "":
                physics_start_epoch = 40
                break
            physics_start_epoch = int(physics_start_epoch)
            if 0 <= physics_start_epoch <= total_epochs:
                break
            else:
                print(f"❌ 物理约束开启轮数必须在0到{total_epochs}之间")
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n❌ 用户取消操作")
            exit(0)
    print(f"\n✓ 训练参数设置完成:")
    print(f"  - 总训练轮数: {total_epochs}")
    print(f"  - 物理约束开启轮数: {physics_start_epoch}")
    return total_epochs, physics_start_epoch
def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='PINO雪崩模拟训练脚本')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['data_driven', 'physics_constrained'], help='训练模式')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--physics-start', type=int, help='物理约束开启轮数')
    parser.add_argument('--non-interactive', action='store_true', help='非交互式运行')
    args = parser.parse_args()
    if args.config:
        config_path = args.config
    else:
        config_path = os.environ.get('PINO_CONFIG_PATH')
        if not config_path:
            base_dir = Path(__file__).parent
            config_path = str(base_dir / "configs" / "config_optimized.yaml")
            print(f"⚠️  未指定配置文件，使用默认路径: {config_path}")
            print(f"   建议使用优化后的配置文件以获得最佳训练效果")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if args.mode:
        selected_mode = args.mode
        print(f"✓ 使用命令行指定的训练模式: {selected_mode}")
    elif args.non_interactive:
        selected_mode = 'physics_constrained'
        print(f"✓ 非交互式模式，使用默认训练模式: {selected_mode}")
    else:
        selected_mode = select_training_mode()
    config['training']['mode'] = selected_mode
    if args.epochs and args.physics_start is not None:
        total_epochs = args.epochs
        physics_start_epoch = args.physics_start
        print(f"✓ 使用命令行指定的训练参数: 总轮数={total_epochs}, 物理约束开启轮数={physics_start_epoch}")
    elif args.non_interactive:
        total_epochs = 200
        physics_start_epoch = 40
        print(f"✓ 非交互式模式，使用默认训练参数: 总轮数={total_epochs}, 物理约束开启轮数={physics_start_epoch}")
    else:
        total_epochs, physics_start_epoch = get_training_parameters()
    config['training']['epochs'] = total_epochs
    config['training']['physics_constrained']['data_only_epochs'] = physics_start_epoch
    original_patience = config['training']['patience']
    print(f"\n⚠️  重要提醒: 早停耐心值已从配置文件读取: {original_patience}")
    print(f"   如需修改，请直接编辑配置文件 configs/config_optimized.yaml")
    print(f"\n📋 配置更新完成:")
    print(f"  - 训练模式: {selected_mode}")
    print(f"  - 总训练轮数: {total_epochs}")
    print(f"  - 物理约束开启轮数: {physics_start_epoch}")
    cuda_available = torch.cuda.is_available()
    use_cuda = config.get('device', {}).get('use_cuda', True) and cuda_available
    force_cpu = config.get('device', {}).get('force_cpu', False)
    if force_cpu:
        device = torch.device('cpu')
        print(f"使用设备: {device} (强制使用CPU)")
    elif use_cuda:
        cuda_device = config.get('device', {}).get('cuda_device', 0)
        device = torch.device(f"cuda:{cuda_device}")
        print(f"使用设备: {device} (CUDA可用)")
    else:
        device = torch.device('cpu')
        if not cuda_available:
            print(f"使用设备: {device} (CUDA不可用，回退到CPU)")
        else:
            print(f"使用设备: {device} (配置为使用CPU)")
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"设置随机种子: {seed}")
    print(f"训练模式: {config['training']['mode']}")
    print(f"按时间对分组: {config['data'].get('group_by_time_pair', True)}")
    print(f"数据目录: {config['paths']['data_dir']}")
    print(f"检查点目录: {config['paths']['checkpoint_dir']}")
    print(f"日志目录: {config['paths']['log_dir']}")
    print(f"结果目录: {config['paths']['results_dir']}")
    trainer = ProgressiveTrainer(config, device)
    trainer.train()
if __name__ == "__main__":
    main()