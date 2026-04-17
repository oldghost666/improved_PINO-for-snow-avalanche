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
import random
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
            raise ValueError("missing required 'data.h5_file_path' parameter in configuration")
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
        is_multi_step = len(target.shape) == 5  
        if is_multi_step:
            return self._compute_multi_step_loss(prediction, target, epoch, height_weight)
        else:
            prediction = prediction.unsqueeze(1)  
            target = target.unsqueeze(1)  
            return self._compute_multi_step_loss(prediction, target, epoch, height_weight)
    def _compute_multi_step_loss(self, prediction: torch.Tensor, target: torch.Tensor, epoch: int = 0, height_weight: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        batch_size, num_steps = prediction.shape[:2]
        total_losses = {}
        step_losses = []
        for step in range(num_steps):
            step_prediction = prediction[:, step]  
            step_target = target[:, step]        
            mse_loss = self.mse_loss(step_prediction, step_target)
            l1_loss = self.l1_loss(step_prediction, step_target)
            huber_loss = self.huber_loss(step_prediction, step_target)
            mask = (step_target.abs() > 0.01).float()
            weighted_mse = torch.sum(mask * (step_prediction - step_target) ** 2) / (torch.sum(mask) + 1e-8)
            height_loss = self.mse_loss(step_prediction[:, 0:1], step_target[:, 0:1])
            velocity_x_loss = self.mse_loss(step_prediction[:, 1:2], step_target[:, 1:2])
            velocity_y_loss = self.mse_loss(step_prediction[:, 2:3], step_target[:, 2:3])
            velocity_loss = (velocity_x_loss + velocity_y_loss) / 2
            if height_weight is not None:
                height_mse_per_pixel = (step_prediction[:, 0:1] - step_target[:, 0:1]) ** 2
                if height_weight.dim() == 3:
                    height_weight_expanded = height_weight.unsqueeze(1)
                else:
                    height_weight_expanded = height_weight
                weighted_height_loss = torch.mean(height_mse_per_pixel * height_weight_expanded)
                enhanced_height_loss = weighted_height_loss
            else:
                enhanced_height_loss = height_loss
            relative_error = torch.mean(torch.abs(step_prediction - step_target) / (torch.abs(step_target) + 1e-6))
            step_total_loss = (
                self.mse_weight * mse_loss + 
                self.weighted_mse_weight * weighted_mse + 
                self.l1_weight * l1_loss + 
                self.height_weight * enhanced_height_loss +
                self.velocity_weight * velocity_loss +
                self.relative_error_weight * relative_error
            )
            step_loss_dict = {
                : step_total_loss,
                : mse_loss,
                : weighted_mse,
                : l1_loss,
                : huber_loss,
                : height_loss,
                : enhanced_height_loss,
                : velocity_loss,
                : relative_error
            }
            step_losses.append(step_loss_dict)
            for key, value in step_loss_dict.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value
        avg_losses = {key: value / num_steps for key, value in total_losses.items()}
        avg_losses['num_steps'] = num_steps
        avg_losses['step_losses'] = step_losses
        return avg_losses
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
        self.enable_memory_monitoring = True
        self.memory_cleanup_interval = 10  
    def monitor_gpu_memory(self, stage: str = ""):
        if torch.cuda.is_available() and self.enable_memory_monitoring:
            allocated = torch.cuda.memory_allocated() / 1024**3  
            reserved = torch.cuda.memory_reserved() / 1024**3   
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  
            if allocated > 10.0:  
                self.logger.warning(f"High VRAM usage {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")
            elif stage and "batch" in stage.lower():
                self.logger.debug(f"VRAM status {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
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
        logger.info(f"optimized training started - Log file: {log_file}")
        return logger
    def setup_realtime_visualization(self):
        try:
            self.fig = None
            self.axes = None
            self.lines = None
            self.logger.info("[Check] Loss data logging enabled (no real-time display)")
        except Exception as e:
            self.logger.warning(f"Loss data logging initialization failed: {e}")
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
            self.logger.warning(f"Loss data logging failed: {e}")
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
            fig.suptitle('PINO Avalanche Training Loss Curves', fontsize=16, fontweight='bold')
            epochs = self.detailed_loss_history['epoch']
            ax1 = axes[0, 0]
            ax1.set_title('Total Loss (Train vs Val)', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.plot(epochs, self.detailed_loss_history['train_total'], 'b-', label='Train Loss', linewidth=2)
            ax1.plot(epochs, self.detailed_loss_history['val_total'], 'r-', label='Val Loss', linewidth=2)
            ax1.legend()
            ax2 = axes[0, 1]
            ax2.set_title('Training Loss Components', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            ax2.plot(epochs, self.detailed_loss_history['data_loss'], 'g-', label='Data Loss', linewidth=2)
            ax2.plot(epochs, self.detailed_loss_history['physics_loss'], 'm-', label='Physics Loss', linewidth=2)
            ax2.plot(epochs, self.detailed_loss_history['boundary_loss'], 'c-', label='Boundary Loss', linewidth=2)
            ax2.legend()
            ax3 = axes[1, 0]
            ax3.set_title('Learning Rate', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            ax3.plot(epochs, self.detailed_loss_history['learning_rate'], 'orange', label='Learning rate', linewidth=2)
            ax3.legend()
            ax4 = axes[1, 1]
            ax4.set_title('Loss Weights', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Weight')
            ax4.grid(True, alpha=0.3)
            ax4.plot(epochs, self.detailed_loss_history['data_weight'], 'g--', label='Data Weight', linewidth=2)
            ax4.plot(epochs, self.detailed_loss_history['physics_weight'], 'm--', label='Physics Weight', linewidth=2)
            ax4.plot(epochs, self.detailed_loss_history['boundary_weight'], 'c--', label='Boundary Weight', linewidth=2)
            ax4.legend()
            plt.tight_layout()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Loss curves saved to: {save_path}")
            import json
            data_path = str(save_path).replace('.png', '_data.json')
            with open(data_path, 'w', encoding='utf-8') as f:
                serializable_history = {}
                for key, values in self.detailed_loss_history.items():
                    serializable_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Loss data saved to: {data_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save loss curves: {e}")
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
        self.logger.info(f"[Config] Physics module selection: {physics_module_type}")
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
                self.logger.info("[OK] Successfully initialized DimensionlessPhysicsLoss")
                self.logger.info("[Stats] Features: Dimensionless processing based on physical scales")
                self.logger.info("[Target] Advantages: Scale correction, normalization, and consistency")
            except Exception as e:
                self.logger.warning(f"[Error] Dimensionless physics loss module initialization failed: {e}")
                self.logger.warning("[Update] Falling back to improved physics loss module")
                physics_module_type = 'improved'
        if physics_module_type == 'improved':
            raise RuntimeError("Dimensionless physics loss module failed, check config and dependencies")
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
        self.logger.info(f"Model parameters count: {sum(p.numel() for p in model.parameters()):,}")
        return model, physics_loss, optimizer, scheduler
    def create_data_loaders(self) -> Tuple:
        if 'group_by_time_scale' in self.config['data']:
            self.logger.warning("[Warning] Deprecated 'group_by_time_scale' detected, use 'group_by_time_pair'")
            self.logger.info("[Check] Using 'group_by_time_pair' strategy")
        train_tile_ids = self.config['data']['train_tile_ids']
        val_tile_ids = self.config['data']['val_tile_ids']
        multi_step_config = self.config['data'].get('multi_step_sampling', {})
        sampling_mode = multi_step_config.get('mode', 'single_step')
        sequence_length = self.config['data'].get('sequence_length', 1)
        prediction_steps = multi_step_config.get('prediction_steps', 1)
        sliding_window_step = multi_step_config.get('sliding_window_step', 1)
        train_dataset = ImprovedAvalancheDataset(
            h5_file_path=self.config['data']['h5_file_path'],
            tile_ids=train_tile_ids,
            sequence_length=sequence_length,
            prediction_steps=prediction_steps,
            normalize=self.config['data']['normalize'],
            dx=self.config['data']['dx'],
            dy=self.config['data']['dy'],
            dt=self.config['data']['dt'],
            boundary_condition=self.config['data']['boundary_condition'],
            sliding_window_step=sliding_window_step,
            sampling_mode=sampling_mode
        )
        val_dataset = ImprovedAvalancheDataset(
            h5_file_path=self.config['data']['h5_file_path'],
            tile_ids=self.config['data']['val_tile_ids'],
            sequence_length=sequence_length,
            prediction_steps=prediction_steps,
            normalize=self.config['data']['normalize'],
            dx=self.config['data']['dx'],
            dy=self.config['data']['dy'],
            dt=self.config['data']['dt'],
            boundary_condition=self.config['data']['boundary_condition'],
            sliding_window_step=sliding_window_step,
            sampling_mode=sampling_mode
        )
        mode_name = "Multi-step autoregressive" if sampling_mode == 'multi_step' else "Single-step prediction"
        if sampling_mode == 'multi_step':
            print(f"validation set sampling strategy: {mode_name}({sequence_length}frame sliding window,{prediction_steps}stepsprediction)")
        else:
            print(f"validation set sampling strategy: {mode_name}")
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
            group_by_time_pair=True  
        )
        self.logger.info(f"Training set size: {len(train_dataset)}")
        self.logger.info(f"Validation set size: {len(val_dataset)}")
        self.logger.info(f"trainingdata loading strategy: Multi-step autoregressive(Time pairgrouping)")
        self.logger.info(f"validation data loading strategy: Multi-step autoregressive(Time pairgrouping)")
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
            data_weight = 1.0
            physics_weight = mode_config.get('physics_weight', 0.8)
            boundary_weight = mode_config.get('boundary_condition_weight', 0.3)
            initial_condition_weight = mode_config.get('initial_condition_weight', 0.5)
            if epoch < data_only_epochs:
                self.logger.info(f"Epoch {epoch+1}: Pure data training phase (first{data_only_epochs}epochs)")
                return {
                    : 1.0,
                    : 0.0,
                    : 0.0,
                    : 0.0
                }
            else:
                multi_step_config = self.config.get('data', {}).get('multi_step_sampling', {})
                if multi_step_config.get('enable', False):
                    weight_scheduling = self.config['training'].get('weight_scheduling', {})
                    if weight_scheduling.get('enable', False):
                        scheduling_type = weight_scheduling.get('type', 'linear')
                        start_epoch = weight_scheduling.get('start_epoch', data_only_epochs)
                        end_epoch = weight_scheduling.get('end_epoch', 100)
                        if epoch >= start_epoch and epoch <= end_epoch:
                            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
                            if scheduling_type == 'linear':
                                data_weight = data_weight * (1 - progress * 0.5)  
                                physics_weight = physics_weight * (0.5 + progress * 0.5)  
                                boundary_weight = boundary_weight * (0.5 + progress * 0.5)  
                                initial_condition_weight = initial_condition_weight * (0.5 + progress * 0.5)  
                            elif scheduling_type == 'exponential':
                                import math
                                data_weight = data_weight * math.exp(-progress * 0.7)
                                physics_weight = physics_weight * (0.3 + 0.7 * (1 - math.exp(-progress * 2)))
                                boundary_weight = boundary_weight * (0.3 + 0.7 * (1 - math.exp(-progress * 2)))
                                initial_condition_weight = initial_condition_weight * (0.3 + 0.7 * (1 - math.exp(-progress * 2)))
                            elif scheduling_type == 'cosine':
                                import math
                                data_weight = data_weight * (0.5 + 0.5 * math.cos(progress * math.pi * 0.5))
                                physics_weight = physics_weight * (0.5 + 0.5 * (1 - math.cos(progress * math.pi * 0.5)))
                                boundary_weight = boundary_weight * (0.5 + 0.5 * (1 - math.cos(progress * math.pi * 0.5)))
                                initial_condition_weight = initial_condition_weight * (0.5 + 0.5 * (1 - math.cos(progress * math.pi * 0.5)))
                if epoch == data_only_epochs:
                    self.logger.info(f"Epoch {epoch+1}: Physics-constrained training enabled!")
                total_weight = data_weight + physics_weight + boundary_weight + initial_condition_weight
                if total_weight > 0:
                    norm_factor = total_weight / (data_weight + physics_weight + boundary_weight + initial_condition_weight)
                    data_weight = max(0.0, data_weight)
                    physics_weight = max(0.0, physics_weight)
                    boundary_weight = max(0.0, boundary_weight)
                    initial_condition_weight = max(0.0, initial_condition_weight)
                return {
                    : data_weight,
                    : physics_weight,
                    : boundary_weight,
                    : initial_condition_weight
                }
        else:
            self.logger.warning(f"Unknown training mode: {training_mode},using default weights")
            return {
                : 1.0,
                : 0.0,
                : 0.0,
                : 0.0
            }
    def get_scheduled_sampling_ratio(self, epoch: int) -> float:
        ss_config = self.config['training'].get('scheduled_sampling', {})
        if not ss_config.get('enabled', False):
            return 1.0  
        start_ratio = ss_config.get('start_ratio', 1.0)  
        end_ratio = ss_config.get('end_ratio', 0.5)      
        decay_epochs = ss_config.get('decay_epochs', 100)  
        decay_type = ss_config.get('decay_type', 'linear')  
        relative_epoch = max(0, min(epoch, decay_epochs))
        if decay_type == 'linear':
            ratio = start_ratio - (start_ratio - end_ratio) * (relative_epoch / decay_epochs)
        elif decay_type == 'exponential':
            decay_rate = ss_config.get('decay_rate', 0.95)
            ratio = max(end_ratio, start_ratio * (decay_rate ** relative_epoch))
        elif decay_type == 'cosine':
            import math
            ratio = end_ratio + (start_ratio - end_ratio) * (1 + math.cos(math.pi * relative_epoch / decay_epochs)) / 2
        else:
            ratio = start_ratio - (start_ratio - end_ratio) * (relative_epoch / decay_epochs)
        return max(0.0, min(1.0, ratio))
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        weights = self.get_current_weights(epoch)
        scheduled_sampling_ratio = self.get_scheduled_sampling_ratio(epoch)
        enable_physics_loss = self.config['physics'].get('enable_physics_loss', True)
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        total_boundary_loss = 0.0
        total_initial_condition_loss = 0.0
        num_batches = 0
        memory_cleanup_interval = 20
        for batch_idx, batch_data in enumerate(self.train_loader):
            input_dynamic = batch_data['input_dynamic'].to(self.device)
            input_static = batch_data['input_static'].to(self.device)
            input_physics = batch_data['input_physics'].to(self.device)
            if 'target_seq' in batch_data:
                targets = batch_data['target_seq'].to(self.device)  
            else:
                targets = batch_data['target'].to(self.device).unsqueeze(1)  
            num_time_steps = targets.shape[1]
            height_weight = batch_data.get('height_weight', None)
            if height_weight is not None:
                height_weight = height_weight.to(self.device)
            batch_data_loss = 0.0
            batch_physics_loss = 0.0
            batch_boundary_loss = 0.0
            batch_initial_condition_loss = 0.0
            current_input_dynamic = input_dynamic
            current_input = torch.cat([current_input_dynamic, input_static, input_physics], dim=1)
            if batch_idx % 20 == 0:
                self.monitor_gpu_memory(f"Epoch {epoch}, Batch {batch_idx}")
                self.cleanup_memory()
            self.optimizer.zero_grad()
            for t in range(num_time_steps):
                predictions = self.model(current_input)
                current_targets = targets[:, t]  
                loss_dict = self.optimized_loss(predictions, current_targets, epoch, height_weight)
                step_data_loss = loss_dict['total_loss']
                step_physics_loss = torch.tensor(0.0, device=self.device)
                if weights['physics_weight'] > 0 and enable_physics_loss:
                    try:
                        predictions_for_physics = predictions.clone().requires_grad_(True)
                        step_physics_loss = self.physics_loss.compute_physics_loss(
                            current_input, predictions_for_physics,
                            self.train_dataset.normalization_stats,
                            self.train_dataset.normalization_stats,
                            step=t,  
                            model=self.model
                        )
                        del predictions_for_physics
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    except Exception as e:
                        self.logger.warning(f"time steps{t}Physics loss calculation failed: {e}")
                        step_physics_loss = torch.tensor(0.0, device=self.device)
                step_boundary_loss = torch.tensor(0.0, device=self.device)
                if weights['boundary_weight'] > 0 and enable_physics_loss:
                    try:
                        step_boundary_loss = self.physics_loss.compute_boundary_loss(
                            self.model, current_input,
                            self.train_dataset.normalization_stats,
                            self.train_dataset.normalization_stats
                        )
                    except Exception as e:
                        self.logger.warning(f"time steps{t}Boundary loss calculation failed: {e}")
                        step_boundary_loss = torch.tensor(0.0, device=self.device)
                step_initial_condition_loss = torch.tensor(0.0, device=self.device)
                if t == 0 and weights['initial_condition_weight'] > 0 and enable_physics_loss:
                    try:
                        step_initial_condition_loss = self.physics_loss.compute_initial_condition_loss(
                            self.model, current_input,
                            self.train_dataset.normalization_stats,
                            self.train_dataset.normalization_stats
                        )
                    except Exception as e:
                        self.logger.warning(f"Initial condition loss calculation failed: {e}")
                        step_initial_condition_loss = torch.tensor(0.0, device=self.device)
                step_loss = (
                    weights['data_weight'] * step_data_loss +
                    weights['physics_weight'] * step_physics_loss +
                    weights['boundary_weight'] * step_boundary_loss +
                    weights['initial_condition_weight'] * step_initial_condition_loss
                )
                batch_data_loss += step_data_loss / num_time_steps
                batch_physics_loss += step_physics_loss / num_time_steps
                batch_boundary_loss += step_boundary_loss / num_time_steps
                batch_initial_condition_loss += step_initial_condition_loss / num_time_steps
                step_loss.backward(retain_graph=True if t < num_time_steps - 1 else False)
                if t < num_time_steps - 1:
                    if random.random() < scheduled_sampling_ratio:
                        next_dynamic_fields = targets[:, t]  
                    else:
                        next_dynamic_fields = predictions  
                    current_input_dynamic = torch.cat([
                        next_dynamic_fields,  
                        input_dynamic[:, 3:6]  
                    ], dim=1)  
                    current_input = torch.cat([current_input_dynamic, input_static, input_physics], dim=1)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['training']['gradient_clip']
            )
            self.optimizer.step()
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            total_batch_loss = batch_data_loss + batch_physics_loss + batch_boundary_loss + batch_initial_condition_loss
            total_loss += total_batch_loss.item()
            total_data_loss += batch_data_loss.item()
            total_physics_loss += batch_physics_loss.item()
            total_boundary_loss += batch_boundary_loss.item()
            total_initial_condition_loss += batch_initial_condition_loss.item()
            num_batches += 1
            if batch_idx % self.memory_cleanup_interval == 0:
                self.monitor_gpu_memory(f"Epoch {epoch}, Batch {batch_idx}")
                self.cleanup_memory()
            if batch_idx % 10 == 0:
                self.logger.info(
                )
                self.monitor_gpu_memory(f"Progress Report - Epoch {epoch}, Batch {batch_idx}")
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
            : current_weights['boundary_weight'],
            : scheduled_sampling_ratio
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
                is_multi_step = 'target_seq' in batch_data
                if is_multi_step:
                    targets = batch_data['target_seq'].to(self.device)  
                    num_time_steps = targets.shape[1]
                else:
                    targets = batch_data['target'].to(self.device)  
                    num_time_steps = 1
                height_weight = batch_data.get('height_weight', None)
                if height_weight is not None:
                    height_weight = height_weight.to(self.device)
                if is_multi_step and num_time_steps > 1:
                    batch_data_loss = 0.0
                    batch_physics_loss = 0.0
                    batch_boundary_loss = 0.0
                    batch_initial_condition_loss = 0.0
                    batch_mse = 0.0
                    batch_mae = 0.0
                    current_input_dynamic = input_dynamic
                    current_input = torch.cat([current_input_dynamic, input_static, input_physics], dim=1)
                    for t in range(num_time_steps):
                        predictions = self.model(current_input)
                        current_targets = targets[:, t]  
                        loss_dict = self.optimized_loss(predictions, current_targets, epoch, height_weight)
                        step_data_loss = loss_dict['total_loss']
                        step_mse = loss_dict['mse_loss']
                        step_mae = loss_dict['l1_loss']
                        step_physics_loss = torch.tensor(0.0, device=self.device)
                        if weights['physics_weight'] > 0:
                            try:
                                with torch.enable_grad():
                                    predictions_for_physics = self.model(current_input)
                                    step_physics_loss = self.physics_loss.compute_physics_loss(
                                        current_input, predictions_for_physics,
                                        self.val_dataset.normalization_stats,
                                        self.val_dataset.normalization_stats,
                                        step=t,  
                                        model=self.model
                                    )
                                    del predictions_for_physics
                                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            except Exception as e:
                                step_physics_loss = torch.tensor(0.0, device=self.device)
                        step_initial_condition_loss = torch.tensor(0.0, device=self.device)
                        if (weights['initial_condition_weight'] > 0 and 
                            hasattr(self.physics_loss, 'enable_initial_condition_loss') and
                            self.physics_loss.enable_initial_condition_loss):
                            try:
                                step_initial_condition_loss = self.physics_loss.compute_initial_condition_loss(
                                    self.model, current_input,
                                    self.val_dataset.normalization_stats,
                                    self.val_dataset.normalization_stats
                                )
                            except Exception as e:
                                step_initial_condition_loss = torch.tensor(0.0, device=self.device)
                        step_boundary_loss = torch.tensor(0.0, device=self.device)
                        if weights['boundary_weight'] > 0:
                            try:
                                step_boundary_loss = self.physics_loss.compute_boundary_loss(
                                    self.model, current_input,
                                    self.val_dataset.normalization_stats,
                                    self.val_dataset.normalization_stats
                                )
                            except Exception as e:
                                step_boundary_loss = torch.tensor(0.0, device=self.device)
                        batch_data_loss += step_data_loss / num_time_steps
                        batch_physics_loss += step_physics_loss / num_time_steps
                        batch_boundary_loss += step_boundary_loss / num_time_steps
                        batch_initial_condition_loss += step_initial_condition_loss / num_time_steps
                        batch_mse += step_mse / num_time_steps
                        batch_mae += step_mae / num_time_steps
                        if t < num_time_steps - 1:
                            next_dynamic_fields = targets[:, t]  
                            current_input_dynamic = torch.cat([
                                next_dynamic_fields,  
                                input_dynamic[:, 3:6]  
                            ], dim=1)  
                            current_input = torch.cat([current_input_dynamic, input_static, input_physics], dim=1)
                    total_batch_loss = (
                        weights['data_weight'] * batch_data_loss +
                        weights['physics_weight'] * batch_physics_loss +
                        weights['boundary_weight'] * batch_boundary_loss +
                        weights['initial_condition_weight'] * batch_initial_condition_loss
                    )
                    total_loss += total_batch_loss.item()
                    total_data_loss += batch_data_loss.item()
                    total_physics_loss += batch_physics_loss.item()
                    total_boundary_loss += batch_boundary_loss.item()
                    total_initial_condition_loss += batch_initial_condition_loss.item()
                    total_mse += batch_mse.item()
                    total_mae += batch_mae.item()
                else:
                    if is_multi_step:
                        targets_for_loss = targets[:, 0]  
                    else:
                        targets_for_loss = targets
                    inputs = torch.cat([input_dynamic, input_static, input_physics], dim=1)
                    predictions = self.model(inputs)
                    loss_dict = self.optimized_loss(predictions, targets_for_loss, epoch, height_weight)
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
            self.logger.info(f"Best model saved: {best_path}")
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    def train(self):
        self.logger.info("Starting progressive training...")
        self.logger.info(f"Total training epochs: {self.config['training']['epochs']}")
        self.logger.info(f"Early stopping patience: {self.config['training']['patience']}")
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
                self.logger.info(f"Early stopping triggered at epoch {epoch+1} ")
                break
        final_path = Path(self.config['paths']['checkpoint_dir']) / 'final_model_optimized.pth'
        torch.save({
            : self.model.state_dict(),
            : self.config,
            : self.training_history
        }, final_path)
        if self.enable_realtime_plot:
            self.save_loss_curves()
        self.logger.info(f"Training completed, final model saved to: {final_path}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
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