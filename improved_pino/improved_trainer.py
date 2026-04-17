import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict
from improved_physics_dimensionless import DimensionlessPhysicsLoss, create_dimensionless_physics_loss
from improved_model import ImprovedPINO
from global_data_config import GlobalDataConfig
from improved_dataset import ImprovedAvalancheDataset, create_improved_dataloader
class OptimizedLossFunction(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        data_config = config.get('data', {})
        if 'h5_file_path' not in data_config:
            raise ValueError("配置文件中缺少必需的'data.h5_file_path'参数")
        self.h5_file_path = data_config['h5_file_path']
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss()
        self.mse_weight = 0.4
        self.weighted_mse_weight = 0.3
        self.l1_weight = 0.1
        self.height_weight = 0.3  
        self.velocity_weight = 0.1  
        self.relative_error_weight = 0.05  
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, epoch: int = 0, height_weight: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        prediction = self.clip_outputs(prediction)
        mse_loss = self.mse_loss(prediction, target)
        l1_loss = self.l1_loss(prediction, target)
        huber_loss = self.huber_loss(prediction, target)
        mask = (target.abs() > 0.01).float()  
        weighted_mse = torch.sum(mask * (prediction - target) ** 2) / (torch.sum(mask) + 1e-8)
        height_loss = self.mse_loss(prediction[:, 0:1], target[:, 0:1])
        velocity_x_loss = self.mse_loss(prediction[:, 1:2], target[:, 1:2])
        velocity_y_loss = self.mse_loss(prediction[:, 2:3], target[:, 2:3])
        velocity_loss = (velocity_x_loss + velocity_y_loss) / 2
        if height_weight is not None:
            height_mse_per_pixel = (prediction[:, 0:1] - target[:, 0:1]) ** 2
            if height_weight.dim() == 3:
                height_weight_expanded = height_weight.unsqueeze(1)
            else:
                height_weight_expanded = height_weight
            weighted_height_loss = torch.mean(height_mse_per_pixel * height_weight_expanded)
            enhanced_height_loss = weighted_height_loss
        else:
            enhanced_height_loss = height_loss
        relative_error = torch.mean(torch.abs(prediction - target) / (torch.abs(target) + 1e-6))
        total_loss = (
            self.mse_weight * mse_loss + 
            self.weighted_mse_weight * weighted_mse + 
            self.l1_weight * l1_loss + 
            self.height_weight * enhanced_height_loss +  
            self.velocity_weight * velocity_loss +  
            self.relative_error_weight * relative_error  
        )
        return {
            : total_loss,
            : mse_loss,
            : weighted_mse,
            : l1_loss,
            : huber_loss,
            : height_loss,
            : enhanced_height_loss,
            : velocity_loss,
            : relative_error
        }
    def clip_outputs(self, prediction: torch.Tensor) -> torch.Tensor:
        from global_data_config import get_global_data_config
        global_config = get_global_data_config(self.h5_file_path)
        height_mean, height_std = global_config.get_height_denorm_params()
        height_phys = prediction[:, 0:1] * height_std + height_mean
        velocity_params = global_config.get_velocity_denorm_params()
        vx_mean, vx_std = velocity_params['velocity_x_mean'], velocity_params['velocity_x_std']
        vy_mean, vy_std = velocity_params['velocity_y_mean'], velocity_params['velocity_y_std']
        vx_phys = prediction[:, 1:2] * vx_std + vx_mean
        vy_phys = prediction[:, 2:3] * vy_std + vy_mean
        height_min, height_max = global_config.get_height_physical_range()
        height_buffer = height_max * 1.1  
        vx_min, vx_max = velocity_params['velocity_x_min'], velocity_params['velocity_x_max']
        vy_min, vy_max = velocity_params['velocity_y_min'], velocity_params['velocity_y_max']
        height_clipped_phys = torch.clamp(height_phys, min=0.0, max=height_buffer)  
        vx_clipped_phys = torch.clamp(vx_phys, min=vx_min, max=vx_max)
        vy_clipped_phys = torch.clamp(vy_phys, min=vy_min, max=vy_max)
        height_clipped_norm = (height_clipped_phys - height_mean) / height_std
        vx_clipped_norm = (vx_clipped_phys - vx_mean) / vx_std
        vy_clipped_norm = (vy_clipped_phys - vy_mean) / vy_std
        clipped_prediction = torch.cat([height_clipped_norm, vx_clipped_norm, vy_clipped_norm], dim=1)
        return clipped_prediction
class ProgressiveTrainer:
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.logger = self.setup_logger()
        self.model, self.physics_loss, self.optimizer, self.scheduler = self.create_model_and_components()
        self.train_loader, self.val_loader, self.train_dataset, self.val_dataset = self.create_data_loaders()
        self.optimized_loss = OptimizedLossFunction(config).to(device)
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            : [],
            : [],
            : []
        }
        self.enable_realtime_plot = config.get('visualization', {}).get('enable_realtime_plot', False)
        self.plot_update_interval = config.get('visualization', {}).get('plot_update_interval', 5)
        self.detailed_loss_history = defaultdict(list)
        self.fig = None
        self.axes = None
        self.lines = {}
        if self.enable_realtime_plot:
            self.setup_realtime_visualization()
    def setup_logger(self) -> logging.Logger:
        log_dir = Path(self.config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'optimized_training_{timestamp}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"优化训练开始 - 日志文件: {log_file}")
        return logger
    def setup_realtime_visualization(self):
        try:
            self.fig = None
            self.axes = None
            self.lines = None
            self.logger.info("✓ 损失数据记录已启用（无实时显示）")
        except Exception as e:
            self.logger.warning(f"损失数据记录初始化失败: {e}")
            self.enable_realtime_plot = False
    def update_realtime_plot(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        if not self.enable_realtime_plot:
            return
        try:
            self.detailed_loss_history['epoch'].append(epoch)
            self.detailed_loss_history['train_total'].append(train_metrics['total_loss'])
            self.detailed_loss_history['val_total'].append(val_metrics['val_loss'])
            self.detailed_loss_history['data_loss'].append(train_metrics['data_loss'])
            self.detailed_loss_history['physics_loss'].append(train_metrics['physics_loss'])
            self.detailed_loss_history['boundary_loss'].append(train_metrics['boundary_loss'])
            self.detailed_loss_history['learning_rate'].append(train_metrics['learning_rate'])
            self.detailed_loss_history['data_weight'].append(train_metrics['data_weight'])
            self.detailed_loss_history['physics_weight'].append(train_metrics['physics_weight'])
            self.detailed_loss_history['boundary_weight'].append(train_metrics['boundary_weight'])
        except Exception as e:
            self.logger.warning(f"记录损失数据失败: {e}")
    def save_loss_curves(self, save_path: str = None):
        if not self.enable_realtime_plot or len(self.detailed_loss_history['epoch']) == 0:
            return
        try:
            if save_path is None:
                log_dir = Path(self.config['paths']['log_dir'])
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                save_path = log_dir / f'training_loss_curves_{timestamp}.png'
            plt.ioff()
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('PINO雪崩模型训练损失曲线', fontsize=16, fontweight='bold')
            epochs = self.detailed_loss_history['epoch']
            ax1 = axes[0, 0]
            ax1.set_title('总损失对比 (训练 vs 验证)', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.plot(epochs, self.detailed_loss_history['train_total'], 'b-', label='训练损失', linewidth=2)
            ax1.plot(epochs, self.detailed_loss_history['val_total'], 'r-', label='验证损失', linewidth=2)
            ax1.legend()
            ax2 = axes[0, 1]
            ax2.set_title('训练损失分量', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            ax2.plot(epochs, self.detailed_loss_history['data_loss'], 'g-', label='数据损失', linewidth=2)
            ax2.plot(epochs, self.detailed_loss_history['physics_loss'], 'm-', label='物理损失', linewidth=2)
            ax2.plot(epochs, self.detailed_loss_history['boundary_loss'], 'c-', label='边界损失', linewidth=2)
            ax2.legend()
            ax3 = axes[1, 0]
            ax3.set_title('学习率变化', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            ax3.plot(epochs, self.detailed_loss_history['learning_rate'], 'orange', label='学习率', linewidth=2)
            ax3.legend()
            ax4 = axes[1, 1]
            ax4.set_title('损失权重变化', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Weight')
            ax4.grid(True, alpha=0.3)
            ax4.plot(epochs, self.detailed_loss_history['data_weight'], 'g--', label='数据权重', linewidth=2)
            ax4.plot(epochs, self.detailed_loss_history['physics_weight'], 'm--', label='物理权重', linewidth=2)
            ax4.plot(epochs, self.detailed_loss_history['boundary_weight'], 'c--', label='边界权重', linewidth=2)
            ax4.legend()
            plt.tight_layout()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"损失曲线图已保存至: {save_path}")
            import json
            data_path = str(save_path).replace('.png', '_data.json')
            with open(data_path, 'w', encoding='utf-8') as f:
                serializable_history = {}
                for key, values in self.detailed_loss_history.items():
                    serializable_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            self.logger.info(f"损失数据已保存至: {data_path}")
        except Exception as e:
            self.logger.warning(f"保存损失曲线失败: {e}")
    def create_model_and_components(self) -> Tuple:
        model = ImprovedPINO(
            modes1=self.config['model']['modes1'],
            modes2=self.config['model']['modes2'],
            width=self.config['model']['width'],
            n_layers=self.config['model']['n_layers'],
            in_channels=self.config['model']['in_channels'],
            out_channels=self.config['model']['out_channels'],
            dropout=self.config['model'].get('dropout', 0.1)
        ).to(self.device)
        physics_module_type = self.config['physics'].get('module_type', 'improved')
        self.logger.info(f"📋 物理模块选择: {physics_module_type}")
        self.logger.info("=" * 60)
        if physics_module_type == 'dimensionless':
            try:
                global_data_config = GlobalDataConfig(
                    h5_file_path=self.config['data']['h5_file_path']
                )
                physics_loss = create_dimensionless_physics_loss(
                    config=self.config,
                    global_data_config=global_data_config,
                    device=self.device
                )
                self.logger.info("✅ 成功初始化: DimensionlessPhysicsLoss (无量纲化物理损失模块)")
                self.logger.info("📊 特性: 基于物理尺度的无量纲化处理，解决量纲不一致问题")
                self.logger.info("🎯 优势: 空间导数物理尺度修正、统计参数特征尺度归一化、浅水方程量纲一致性")
            except Exception as e:
                self.logger.warning(f"❌ 无量纲化物理损失模块初始化失败: {e}")
                self.logger.warning("🔄 自动回退到旧版改进物理损失模块")
                physics_module_type = 'improved'
        if physics_module_type == 'improved':
            raise RuntimeError("无量纲化物理损失模块初始化失败，请检查配置文件和依赖项")
        self.logger.info("=" * 60)
        physics_loss = physics_loss.to(self.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        if self.config['scheduler']['type'] == 'CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=self.config['scheduler']['eta_min']
            )
        elif self.config['scheduler']['type'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=self.config['scheduler']['eta_min']
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=50,
                gamma=0.5
            )
        self.logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        return model, physics_loss, optimizer, scheduler
    def create_data_loaders(self) -> Tuple:
        if 'group_by_time_scale' in self.config['data']:
            self.logger.warning("⚠️ 检测到已弃用的配置字段 'group_by_time_scale'，请使用 'group_by_time_pair' 替代")
            self.logger.info("✓ 自动使用 'group_by_time_pair' 策略")
        train_tile_ids = self.config['data']['train_tile_ids']
        val_tile_ids = self.config['data']['val_tile_ids']
        train_dataset = ImprovedAvalancheDataset(
            h5_file_path=self.config['data']['h5_file_path'],
            tile_ids=train_tile_ids,
            sequence_length=self.config['data']['sequence_length'],
            prediction_steps=self.config['data']['prediction_steps'],
            mode=self.config['data']['mode'],
            normalize=self.config['data']['normalize'],
            dx=self.config['data']['dx'],
            dy=self.config['data']['dy'],
            dt=self.config['data']['dt'],
            boundary_condition=self.config['data']['boundary_condition'],
            group_by_time_scale=self.config['data'].get('group_by_time_pair', True),
            time_scale_tolerance=self.config['data'].get('time_scale_tolerance', 0.1)
        )
        val_dataset = ImprovedAvalancheDataset(
            h5_file_path=self.config['data']['h5_file_path'],
            tile_ids=val_tile_ids,
            sequence_length=self.config['data']['sequence_length'],
            prediction_steps=self.config['data']['prediction_steps'],
            mode=self.config['data']['mode'],
            normalize=self.config['data']['normalize'],
            dx=self.config['data']['dx'],
            dy=self.config['data']['dy'],
            dt=self.config['data']['dt'],
            boundary_condition=self.config['data']['boundary_condition'],
            group_by_time_scale=False,  
            time_scale_tolerance=self.config['data'].get('time_scale_tolerance', 0.1)
        )
        train_loader = create_improved_dataloader(
            dataset=train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training'].get('num_workers', 0),
            group_by_time_pair=self.config['data'].get('group_by_time_pair', True)  
        )
        val_loader = create_improved_dataloader(
            dataset=val_dataset,
            batch_size=self.config['training'].get('val_batch_size', 1),
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 0),
            group_by_time_pair=False  
        )
        self.logger.info(f"训练集大小: {len(train_dataset)}")
        self.logger.info(f"验证集大小: {len(val_dataset)}")
        self.logger.info(f"训练数据加载策略: 时间对分组={self.config['data'].get('group_by_time_pair', True)}")
        return train_loader, val_loader, train_dataset, val_dataset
    def get_current_weights(self, epoch: int) -> Dict[str, float]:
        training_mode = self.config['training']['mode']
        if training_mode == 'data_driven':
            mode_config = self.config['training']['data_driven']
            return {
                : 1.0,
                : mode_config.get('physics_weight', 0.0),
                : mode_config.get('boundary_condition_weight', 0.0),
                : mode_config.get('initial_condition_weight', 0.0)
            }
        elif training_mode == 'physics_constrained':
            mode_config = self.config['training']['physics_constrained']
            data_only_epochs = mode_config.get('data_only_epochs', 0)
            if epoch < data_only_epochs:
                self.logger.info(f"Epoch {epoch+1}: 纯数据训练阶段 (前{data_only_epochs}轮)")
                return {
                    : 1.0,
                    : 0.0,
                    : 0.0,
                    : 0.0
                }
            else:
                if epoch == data_only_epochs:
                    self.logger.info(f"Epoch {epoch+1}: 开启物理约束训练！")
                return {
                    : 1.0,
                    : mode_config.get('physics_weight', 0.8),
                    : mode_config.get('boundary_condition_weight', 0.3),
                    : mode_config.get('initial_condition_weight', 0.5)
                }
        else:
            self.logger.warning(f"未知训练模式: {training_mode}，使用默认权重")
            return {
                : 1.0,
                : 0.0,
                : 0.0,
                : 0.0
            }
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        weights = self.get_current_weights(epoch)
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        total_boundary_loss = 0.0
        total_initial_condition_loss = 0.0
        num_batches = 0
        for batch_idx, batch_data in enumerate(self.train_loader):
            input_dynamic = batch_data['input_dynamic'].to(self.device)
            input_static = batch_data['input_static'].to(self.device)
            input_physics = batch_data['input_physics'].to(self.device)
            targets = batch_data['target'].to(self.device)
            height_weight = batch_data.get('height_weight', None)
            if height_weight is not None:
                height_weight = height_weight.to(self.device)
            inputs = torch.cat([input_dynamic, input_static, input_physics], dim=1)
            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            loss_dict = self.optimized_loss(predictions, targets, epoch, height_weight)
            data_loss = loss_dict['total_loss']
            physics_loss = torch.tensor(0.0, device=self.device)
            if weights['physics_weight'] > 0:
                try:
                    predictions_for_physics = predictions.clone().requires_grad_(True)
                    physics_loss = self.physics_loss.compute_physics_loss(
                        inputs, predictions_for_physics,
                        self.train_dataset.normalization_stats,
                        self.train_dataset.normalization_stats,
                        step=batch_idx,
                        model=self.model
                    )
                    del predictions_for_physics
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except Exception as e:
                    self.logger.warning(f"物理损失计算失败: {e}")
                    physics_loss = torch.tensor(0.0, device=self.device)
            initial_condition_loss = torch.tensor(0.0, device=self.device)
            if (weights['initial_condition_weight'] > 0 and 
                hasattr(self.physics_loss, 'enable_initial_condition_loss') and
                self.physics_loss.enable_initial_condition_loss):
                try:
                    initial_condition_loss = self.physics_loss.compute_initial_condition_loss(
                        self.model, inputs,
                        self.train_dataset.normalization_stats,
                        self.train_dataset.normalization_stats
                    )
                except Exception as e:
                    self.logger.warning(f"初始条件损失计算失败: {e}")
                    initial_condition_loss = torch.tensor(0.0, device=self.device)
            boundary_loss = torch.tensor(0.0, device=self.device)
            if weights['boundary_weight'] > 0:
                try:
                    boundary_loss = self.physics_loss.compute_boundary_loss(
                        self.model, inputs,
                        self.train_dataset.normalization_stats,
                        self.train_dataset.normalization_stats
                    )
                except Exception as e:
                    self.logger.warning(f"边界损失计算失败: {e}")
                    boundary_loss = torch.tensor(0.0, device=self.device)
            total_batch_loss = (
                weights['data_weight'] * data_loss +
                weights['physics_weight'] * physics_loss +
                weights['boundary_weight'] * boundary_loss +
                weights['initial_condition_weight'] * initial_condition_loss
            )
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['training']['gradient_clip']
            )
            self.optimizer.step()
            total_loss += total_batch_loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_initial_condition_loss += initial_condition_loss.item()
            num_batches += 1
            if batch_idx % 10 == 0:
                self.logger.info(
                )
        self.scheduler.step()
        current_weights = self.get_current_weights(epoch)
        return {
            : total_loss / num_batches,
            : total_data_loss / num_batches,
            : total_physics_loss / num_batches,
            : total_initial_condition_loss / num_batches,
            : total_boundary_loss / num_batches,
            : self.optimizer.param_groups[0]['lr'],
            : current_weights['data_weight'],
            : current_weights['physics_weight'],
            : current_weights['initial_condition_weight'],
            : current_weights['boundary_weight']
        }
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        weights = self.get_current_weights(epoch)
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        total_boundary_loss = 0.0
        total_initial_condition_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                input_dynamic = batch_data['input_dynamic'].to(self.device)
                input_static = batch_data['input_static'].to(self.device)
                input_physics = batch_data['input_physics'].to(self.device)
                targets = batch_data['target'].to(self.device)
                height_weight = batch_data.get('height_weight', None)
                if height_weight is not None:
                    height_weight = height_weight.to(self.device)
                inputs = torch.cat([input_dynamic, input_static, input_physics], dim=1)
                predictions = self.model(inputs)
                loss_dict = self.optimized_loss(predictions, targets, epoch, height_weight)
                data_loss = loss_dict['total_loss']
                physics_loss = torch.tensor(0.0, device=self.device)
                if weights['physics_weight'] > 0:
                    try:
                        with torch.enable_grad():
                            predictions_for_physics = self.model(inputs)
                            physics_loss = self.physics_loss.compute_physics_loss(
                                inputs, predictions_for_physics,
                                self.val_dataset.normalization_stats,
                                self.val_dataset.normalization_stats,
                                step=batch_idx,
                                model=self.model
                            )
                            del predictions_for_physics
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    except Exception as e:
                        physics_loss = torch.tensor(0.0, device=self.device)
                initial_condition_loss = torch.tensor(0.0, device=self.device)
                if (weights['initial_condition_weight'] > 0 and 
                    hasattr(self.physics_loss, 'enable_initial_condition_loss') and
                    self.physics_loss.enable_initial_condition_loss):
                    try:
                        initial_condition_loss = self.physics_loss.compute_initial_condition_loss(
                            self.model, inputs,
                            self.val_dataset.normalization_stats,
                            self.val_dataset.normalization_stats
                        )
                    except Exception as e:
                        initial_condition_loss = torch.tensor(0.0, device=self.device)
                boundary_loss = torch.tensor(0.0, device=self.device)
                if weights['boundary_weight'] > 0:
                    try:
                        boundary_loss = self.physics_loss.compute_boundary_loss(
                            self.model, inputs,
                            self.val_dataset.normalization_stats,
                            self.val_dataset.normalization_stats
                        )
                    except Exception as e:
                        boundary_loss = torch.tensor(0.0, device=self.device)
                total_batch_loss = (
                    weights['data_weight'] * data_loss +
                    weights['physics_weight'] * physics_loss +
                    weights['boundary_weight'] * boundary_loss +
                    weights['initial_condition_weight'] * initial_condition_loss
                )
                total_loss += total_batch_loss.item()
                total_data_loss += data_loss.item()
                total_physics_loss += physics_loss.item()
                total_boundary_loss += boundary_loss.item()
                total_initial_condition_loss += initial_condition_loss.item()
                total_mse += loss_dict['mse_loss'].item()
                total_mae += loss_dict['l1_loss'].item()
                num_batches += 1
        return {
            : total_loss / num_batches,
            : total_data_loss / num_batches,
            : total_physics_loss / num_batches,
            : total_initial_condition_loss / num_batches,
            : total_boundary_loss / num_batches,
            : total_mse / num_batches,
            : total_mae / num_batches
        }
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            : epoch,
            : self.model.state_dict(),
            : self.optimizer.state_dict(),
            : self.scheduler.state_dict(),
            : self.best_val_loss,
            : self.training_history,
            : self.config
        }
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型: {best_path}")
        self.logger.info(f"保存检查点: {checkpoint_path}")
    def train(self):
        self.logger.info("开始渐进式训练...")
        self.logger.info(f"总训练轮数: {self.config['training']['epochs']}")
        self.logger.info(f"早停耐心: {self.config['training']['patience']}")
        for epoch in range(self.config['training']['epochs']):
            start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['learning_rate'].append(train_metrics['learning_rate'])
            if self.enable_realtime_plot:
                self.update_realtime_plot(epoch, train_metrics, val_metrics)
            self.scheduler.step()
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            epoch_time = time.time() - start_time
            self.logger.info(
            )
            if self.patience_counter >= self.config['training']['patience']:
                self.logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        final_path = Path(self.config['paths']['checkpoint_dir']) / 'final_model_optimized.pth'
        torch.save({
            : self.model.state_dict(),
            : self.config,
            : self.training_history
        }, final_path)
        if self.enable_realtime_plot:
            self.save_loss_curves()
        self.logger.info(f"训练完成，最终模型保存至: {final_path}")
        self.logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")
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