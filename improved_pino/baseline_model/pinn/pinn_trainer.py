
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib
import matplotlib.font_manager as fm
from scipy.spatial import ConvexHull
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from pinn_model import AvalanchePINN, AvalanchePINNWithPhysics
from pinn_physics_loss import DimensionlessPhysicsLoss, AvalanchePhysicsLoss
class AdvancedWeightScheduler:
    def __init__(
        self,
        total_epochs: int = 300,
        training_mode: str = 'progressive',
        data_loss_weight: float = 1.0,
        physics_loss_weight: float = 1.0,
        boundary_loss_weight: float = 0.2,
        initial_loss_weight: float = 0.5,
        data_only_epochs: int = 20,
        physics_intro_epochs: int = 50,
        initial_physics_weight: float = 0.1,
        final_physics_weight: float = 1.2
        ):
        self.total_epochs = total_epochs
        self.training_mode = training_mode.lower()
        self.data_loss_weight = data_loss_weight
        self.physics_loss_weight = physics_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.initial_loss_weight = initial_loss_weight
        self.data_only_epochs = data_only_epochs
        self.physics_intro_epochs = physics_intro_epochs
        self.initial_physics_weight = initial_physics_weight
        self.final_physics_weight = final_physics_weight
        self._ema_beta = 0.9
        self._ema_data_loss = None
        self._ema_physics_loss = None
        self._eps = 1e-8
    def update_running_losses(self, data_loss: float, physics_loss: float):
        if data_loss is None or physics_loss is None:
            return
        if self._ema_data_loss is None:
            self._ema_data_loss = float(data_loss)
            self._ema_physics_loss = float(physics_loss)
        else:
            b = self._ema_beta
            self._ema_data_loss = b * self._ema_data_loss + (1 - b) * float(data_loss)
            self._ema_physics_loss = b * self._ema_physics_loss + (1 - b) * float(physics_loss)
    def get_weights(self, epoch: int) -> Dict[str, float]:
        if self.training_mode == 'data_driven':
            return self._get_data_driven_weights()
        elif self.training_mode == 'physics_driven':
            return self._get_physics_driven_weights()
        elif self.training_mode == 'joint_training':
            return self._get_joint_training_weights()
        else:  
            return self._get_progressive_weights(epoch)
    def _get_data_driven_weights(self) -> Dict[str, float]:
        return {
            : self.data_loss_weight,
            : 0.0,
            : 0.0,
            : 0.0
        }
    def _get_physics_driven_weights(self) -> Dict[str, float]:
        return {
            : self.data_loss_weight,
            : self.physics_loss_weight,
            : self.boundary_loss_weight,
            : self.initial_loss_weight
        }
    def _get_joint_training_weights(self) -> Dict[str, float]:
        return {
            : self.data_loss_weight if hasattr(self, 'data_loss_weight') else 1.0,
            : self.physics_loss_weight if hasattr(self, 'physics_loss_weight') else 0.8,
            : 0.0,
            : 0.0
        }
    def _get_progressive_weights(self, epoch: int) -> Dict[str, float]:
        weights = {
            : 1.0,
            : 0.0,
            : self.boundary_loss_weight,
            : self.initial_loss_weight
        }
        if epoch < self.data_only_epochs:
            weights['physics'] = self.initial_physics_weight * 0.05
            weights['boundary'] = self.boundary_loss_weight * 0.5
        elif epoch < self.data_only_epochs + self.physics_intro_epochs:
            progress = (epoch - self.data_only_epochs) / self.physics_intro_epochs
            weights['physics'] = self.initial_physics_weight * (0.1 + 0.4 * progress)
        elif epoch < 100:
            weights['physics'] = self.initial_physics_weight * 0.5
        elif epoch < 200:
            weights['physics'] = self.initial_physics_weight
        else:
            weights['physics'] = self.initial_physics_weight * (1 + (epoch - 200) / 50)
        if self._ema_data_loss is not None and self._ema_physics_loss is not None:
            ratio = (self._ema_data_loss + self._eps) / (self._ema_physics_loss + self._eps)
            weights['physics'] = float(np.clip(weights['physics'] * ratio, 0.0, 1.5))
        if epoch < 30:
            weights['boundary'] = self.boundary_loss_weight * 0.5
        elif epoch >= 150:
            weights['boundary'] = self.boundary_loss_weight * 1.5
        return weights
class DynamicBatchScheduler:
    def __init__(
        self,
        initial_batch_size: int = 512,
        min_batch_size: int = 32,
        reduction_factor: float = 0.5,
        reduction_epochs: int = 100
    ):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.reduction_factor = reduction_factor
        self.reduction_epochs = reduction_epochs
        self.current_batch_size = initial_batch_size
    def get_batch_size(self, epoch: int) -> int:
        if epoch > 0 and epoch % self.reduction_epochs == 0:
            new_batch_size = max(
                int(self.current_batch_size * self.reduction_factor),
                self.min_batch_size
            )
            if new_batch_size != self.current_batch_size:
                self.current_batch_size = new_batch_size
                logging.info(f"Epoch {epoch}: reducing batch size to {self.current_batch_size}")
        return self.current_batch_size
class EnhancedAvalanchePINNTrainer:
    def __init__(
        self,
        model: AvalanchePINN,
        physics_loss: AvalanchePhysicsLoss,  
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.model = model.to(device)
        self.physics_loss = physics_loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        train_config = config.get('training', {})
        opt_cfg = config.get('optimizer', {})
        sch_cfg = config.get('scheduler', {})
        self.num_epochs = int(train_config.get('num_epochs', 300))
        lr_cfg = train_config.get('initial_lr', opt_cfg.get('lr', 5e-4))
        wd_cfg = opt_cfg.get('weight_decay', train_config.get('weight_decay', 1e-5))
        try:
            self.learning_rate = float(lr_cfg)
        except Exception:
            self.learning_rate = 5e-4
        try:
            self.weight_decay = float(wd_cfg)
        except Exception:
            self.weight_decay = 1e-5
        self.gradient_clip = float(train_config.get('grad_clip_value', train_config.get('gradient_clip', 1.0)))
        self._optimizer_cfg = opt_cfg
        self._scheduler_cfg = sch_cfg
        training_mode = train_config.get('mode', 'progressive')
        data_loss_weight = float(train_config.get('data_loss_weight', 1.0))
        physics_loss_weight = float(train_config.get('physics_loss_weight', 1.0))
        boundary_loss_weight = float(train_config.get('boundary_loss_weight', 0.2))
        initial_loss_weight = float(train_config.get('initial_loss_weight', 0.5))
        self.weight_scheduler = AdvancedWeightScheduler(
            total_epochs=self.num_epochs,
            training_mode=training_mode,
            data_loss_weight=data_loss_weight,
            physics_loss_weight=physics_loss_weight,
            boundary_loss_weight=boundary_loss_weight,
            initial_loss_weight=initial_loss_weight,
            data_only_epochs=train_config.get('data_only_epochs', 20),
            physics_intro_epochs=train_config.get('physics_intro_epochs', 50),
            initial_physics_weight=train_config.get('initial_physics_weight', 0.3),
            final_physics_weight=train_config.get('final_physics_weight', 1.2)
        )
        chw = train_config.get('channel_loss_weights', {})
        self.height_loss_w = float(chw.get('height', 1.0))
        self.vx_loss_w = float(chw.get('velocity_x', 1.0))
        self.vy_loss_w = float(chw.get('velocity_y', 1.0))
        self.active_region_weight = float(train_config.get('active_region_weight', 1.0))
        self.active_height_threshold = float(self.config.get('data', {}).get('sampling', {}).get('active_height_threshold', 0.01))
        self.height_max_value = float(train_config.get('height_max_value', 50.0))
        self.height_neg_penalty = float(train_config.get('height_neg_penalty', 0.0))
        self.height_over_penalty = float(train_config.get('height_over_penalty', 0.0))
        self.velocity_l2_penalty = float(train_config.get('velocity_l2_penalty', 0.0))
        self.smape_weight = float(train_config.get('smape_weight', 0.0))
        self.height_distribution_align_weight = float(train_config.get('height_distribution_align_weight', 0.0))
        self.height_peak_align_weight = float(train_config.get('height_peak_align_weight', 0.0))
        self.height_peak_percentile = float(train_config.get('height_peak_percentile', 0.95))
        config_batch_size = train_config.get('batch_size', 32)  
        self.batch_scheduler = DynamicBatchScheduler(
            initial_batch_size=train_config.get('initial_batch_size', config_batch_size),
            min_batch_size=train_config.get('min_batch_size', max(config_batch_size // 4, 8)),
            reduction_factor=train_config.get('batch_reduction_factor', 0.8),  
            reduction_epochs=train_config.get('batch_reduction_epochs', 50)  
        )
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.plateau_scheduler = None
        reduce_cfg = self.config.get('training', {})
        if bool(reduce_cfg.get('reduce_on_plateau', False)):
            factor = float(reduce_cfg.get('plateau_factor', 0.5))
            patience = int(reduce_cfg.get('plateau_patience', 10))
            min_lr = float(reduce_cfg.get('plateau_min_lr', 1e-6))
            self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience,
                threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=min_lr, eps=1e-08
            )
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        self.checkpoint_dir = Path(self.config.get('paths', {}).get('checkpoints_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_val_loss = float('inf')
        self.best_model_path = self.checkpoint_dir / 'best_model.pth'
        self.interval_best_val_loss = float('inf')
        self.interval_best_epoch = -1
        self.interval_size = 50  
        self.patience = train_config.get('patience', 100)
        physics_loss_cfg = config.get('physics_loss', {})
        self.max_loss_value = float(physics_loss_cfg.get('max_loss_value', 50.0))
        self.patience_counter = 0
        self.gradient_method = train_config.get('gradient_method', 'autograd')  
        self.gradient_sample_size = train_config.get('gradient_sample_size', 100)  
        self.verify_gradients = train_config.get('verify_gradients', False)  
        self.setup_logging()
        logging.info(f"Enhanced avalanche PINN trainer initialization completed:")
        logging.info(f"  Device: {device}")
        logging.info(f"  Total epochs: {self.num_epochs}")
        logging.info(f"  Learning rate: {self.learning_rate}")
        logging.info(f"  Weight decay: {self.weight_decay}")
        logging.info(f"  Gradient clip: {self.gradient_clip}")
        logging.info(f"  Initial Batch size: {self.batch_scheduler.initial_batch_size}")
        logging.info(f"  Min Batch size: {self.batch_scheduler.min_batch_size}")
        logging.info(f"  Gradient calculation method: {self.gradient_method}")
        if self.gradient_method == 'autograd':
            logging.info(f"  Gradient sample size: {self.gradient_sample_size}")
        logging.info(f"  Gradient flow verification: {self.verify_gradients}")
    def _setup_optimizer(self) -> optim.Optimizer:
        opt_type = str(self._optimizer_cfg.get('type', 'AdamW')).lower()
        betas_cfg = self._optimizer_cfg.get('betas', (0.9, 0.999))
        try:
            betas = tuple(float(b) for b in betas_cfg)
        except Exception:
            betas = (0.9, 0.999)
        eps = float(self._optimizer_cfg.get('eps', 1e-8))
        if opt_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=betas,
                eps=eps
            )
        else:
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=betas,
                eps=eps
            )
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        sch_type = str(self._scheduler_cfg.get('type', 'CosineAnnealingWarmRestarts'))
        if sch_type == 'CosineAnnealingLR':
            T_max = int(self._scheduler_cfg.get('T_max', max(1, self.num_epochs)))
            eta_min = float(self._scheduler_cfg.get('eta_min', 1e-6))
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
        else:
            T_0 = int(self._scheduler_cfg.get('T_0', 50))
            T_mult = int(self._scheduler_cfg.get('T_mult', 2))
            eta_min = float(self._scheduler_cfg.get('eta_min', 1e-6))
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
    def setup_logging(self):
        log_dir = Path(self.config.get('paths', {}).get('log_dir', './logs'))
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'enhanced_training_{time.strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )
    def _check_numerical_stability(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logging.warning(f"Detected{name}NaN or Inf values in, using replacement values")
            tensor = torch.where(
                torch.isnan(tensor) | torch.isinf(tensor),
                torch.tensor(0.1, device=tensor.device, dtype=tensor.dtype),
                tensor
            )
        return tensor
    def _check_gradients(self) -> bool:
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return False
        return True
    def _compute_autograd_spatial_gradients(
        self,
        X_batch: torch.Tensor,
        sample_indices: torch.Tensor,
        sample_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        if sample_size is not None and sample_indices.shape[0] > sample_size:
            perm = torch.randperm(sample_indices.shape[0], device=self.device)[:sample_size]
            sample_indices = sample_indices[perm]
        b_idx = sample_indices[:, 0]
        y_idx = sample_indices[:, 1] 
        x_idx = sample_indices[:, 2]
        x_coords = X_batch[b_idx, 3, y_idx, x_idx].unsqueeze(1)  
        y_coords = X_batch[b_idx, 4, y_idx, x_idx].unsqueeze(1)  
        other_features = X_batch[b_idx, :, y_idx, x_idx]  
        input_features = other_features.clone()
        input_features[:, 3] = x_coords.squeeze(1)
        input_features[:, 4] = y_coords.squeeze(1)
        x_coords.requires_grad_(True)
        y_coords.requires_grad_(True)
        input_features[:, 3] = x_coords.squeeze(1)
        input_features[:, 4] = y_coords.squeeze(1)
        predictions = self.model.forward_points(input_features)  
        h_pred = predictions[:, 0]
        vx_pred = predictions[:, 1]
        vy_pred = predictions[:, 2]
        dh_dx = torch.autograd.grad(
            outputs=h_pred.sum(), inputs=x_coords,
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        dvx_dx = torch.autograd.grad(
            outputs=vx_pred.sum(), inputs=x_coords,
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        dvy_dx = torch.autograd.grad(
            outputs=vy_pred.sum(), inputs=x_coords,
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        dh_dy = torch.autograd.grad(
            outputs=h_pred.sum(), inputs=y_coords,
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        dvx_dy = torch.autograd.grad(
            outputs=vx_pred.sum(), inputs=y_coords,
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        dvy_dy = torch.autograd.grad(
            outputs=vy_pred.sum(), inputs=y_coords,
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        if dh_dx is None:
            dh_dx = torch.zeros_like(predictions[:, 0:1])
        if dvx_dx is None:
            dvx_dx = torch.zeros_like(predictions[:, 0:1])
        if dvy_dx is None:
            dvy_dx = torch.zeros_like(predictions[:, 0:1])
        if dh_dy is None:
            dh_dy = torch.zeros_like(predictions[:, 0:1])
        if dvx_dy is None:
            dvx_dy = torch.zeros_like(predictions[:, 0:1])
        if dvy_dy is None:
            dvy_dy = torch.zeros_like(predictions[:, 0:1])
        dx_phys = float(getattr(self.physics_loss, 'dx', 1.0))
        dy_phys = float(getattr(self.physics_loss, 'dy', 1.0))
        norm_stats = self._get_norm_stats()
        h_stats = norm_stats.get('height', {'method': 'standard', 'std': 1.0})
        vx_stats = norm_stats.get('velocity_x', {'method': 'standard', 'std': 10.0})
        vy_stats = norm_stats.get('velocity_y', {'method': 'standard', 'std': 10.0})
        if h_stats.get('method') == 'standard':
            h_scale = h_stats.get('std', 1.0)
        else:
            h_scale = h_stats.get('max', 1.0) - h_stats.get('min', 0.0)
        if vx_stats.get('method') == 'standard':
            vx_scale = vx_stats.get('std', 10.0)
        else:
            vx_scale = vx_stats.get('max', 50.0) - vx_stats.get('min', -50.0)
        if vy_stats.get('method') == 'standard':
            vy_scale = vy_stats.get('std', 10.0)
        else:
            vy_scale = vy_stats.get('max', 50.0) - vy_stats.get('min', -50.0)
        gradients = {
            : self._denormalize_tensor(predictions[:, 0:1], 'height'),
            : self._denormalize_tensor(predictions[:, 1:2], 'velocity_x'),
            : self._denormalize_tensor(predictions[:, 2:3], 'velocity_y'),
            : dh_dx * h_scale / dx_phys if dh_dx is not None else torch.zeros_like(predictions[:, 0:1]),
            : dh_dy * h_scale / dy_phys if dh_dy is not None else torch.zeros_like(predictions[:, 0:1]),
            : dvx_dx * vx_scale / dx_phys if dvx_dx is not None else torch.zeros_like(predictions[:, 0:1]),
            : dvx_dy * vx_scale / dy_phys if dvx_dy is not None else torch.zeros_like(predictions[:, 0:1]),
            : dvy_dx * vy_scale / dx_phys if dvy_dx is not None else torch.zeros_like(predictions[:, 0:1]),
            : dvy_dy * vy_scale / dy_phys if dvy_dy is not None else torch.zeros_like(predictions[:, 0:1]),
            : sample_indices  
        }
        return gradients
    def _verify_gradient_flow(self, loss: torch.Tensor) -> Dict[str, Any]:
        loss.backward(retain_graph=True)
        total_norm = 0.0
        zero_grad_count = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if param_norm.item() == 0:
                    zero_grad_count += 1
            else:
                zero_grad_count += 1
            total_params += 1
        total_norm = total_norm ** (1. / 2)
        return {
            : total_norm,
            : zero_grad_count,
            : total_params,
            : zero_grad_count < total_params * 0.5  
        }
    def _get_norm_stats(self) -> Dict[str, Any]:
        try:
            return getattr(self.train_loader.dataset, 'norm_stats', {})
        except Exception:
            return {}
    def _denormalize_tensor(self, tensor: torch.Tensor, var_name: str) -> torch.Tensor:
        if var_name in ['x', 'y', 'time']:
            B = tensor.shape[0]
            H = tensor.shape[2] if tensor.dim() == 4 else 1
            W = tensor.shape[3] if tensor.dim() == 4 else 1
            if var_name == 'x':
                x_scale = W * float(getattr(self.physics_loss, 'dx', 1.0))
                return tensor * x_scale
            elif var_name == 'y':
                y_scale = H * float(getattr(self.physics_loss, 'dy', 1.0))
                return tensor * y_scale
            else:  
                try:
                    time_scale = float(getattr(self.physics_loss, 'dt', 1.0))
                except Exception:
                    time_scale = 1.0
                return tensor * time_scale
        stats = self._get_norm_stats()
        if var_name not in stats:
            return tensor
        s = stats[var_name]
        method = s.get('method', 'standard')
        if method == 'standard':
            mean = torch.tensor(s.get('mean', 0.0), device=tensor.device, dtype=tensor.dtype)
            std = torch.tensor(s.get('std', 1.0), device=tensor.device, dtype=tensor.dtype)
            return tensor * (std + 1e-8) + mean
        elif method == 'minmax':
            vmin = torch.tensor(s.get('min', 0.0), device=tensor.device, dtype=tensor.dtype)
            vmax = torch.tensor(s.get('max', 1.0), device=tensor.device, dtype=tensor.dtype)
            return tensor * (vmax - vmin + 1e-8) + vmin
        else:
            return tensor
    def _assemble_physics_inputs(
        self,
        X_batch: torch.Tensor,
        predictions_phys: Dict[str, torch.Tensor],
        y_batch: Optional[torch.Tensor] = None,
        include_time_and_coords: bool = True,
        sample_meta: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        B = X_batch.shape[0]
        H = predictions_phys['h'].shape[2]
        W = predictions_phys['h'].shape[3]
        dem_norm = X_batch[:, 6:7, :, :]
        dzdx_norm = X_batch[:, 7:8, :, :]
        dzdy_norm = X_batch[:, 8:9, :, :]
        dem = self._denormalize_tensor(dem_norm, 'dem')
        dz_dx = self._denormalize_tensor(dzdx_norm, 'dzdx')
        dz_dy = self._denormalize_tensor(dzdy_norm, 'dzdy')
        boundary_mask, initial_mask = self._generate_boundary_and_initial_masks(X_batch, predictions_phys['h'].shape)
        inputs = {
            : dem,
            : dz_dx,
            : dz_dy,
            : boundary_mask,
            : initial_mask
        }
        if include_time_and_coords:
            try:
                x_norm = X_batch[:, 3:4, :, :]
                y_norm = X_batch[:, 4:5, :, :]
            except Exception:
                x_norm = torch.zeros(B, 1, H, W, device=self.device)
                y_norm = torch.zeros(B, 1, H, W, device=self.device)
            x_phys = self._denormalize_tensor(x_norm, 'x')
            y_phys = self._denormalize_tensor(y_norm, 'y')
            coords_4d = torch.cat([x_phys, y_phys], dim=1)
            inputs['coords'] = coords_4d  
            try:
                t_norm = X_batch[:, 5:6, :, :]
            except Exception:
                t_norm = torch.zeros(B, 1, H, W, device=self.device)
            time_scale = torch.full((B, 1, 1, 1), 200.0, device=self.device, dtype=t_norm.dtype)
            t_phys = t_norm * time_scale
            inputs['t'] = t_phys
            if not hasattr(self, '_time_scale_logged'):
                try:
                    ts_min = float(time_scale.min().item())
                    ts_max = float(time_scale.max().item())
                    logging.info(f"Time scale range: [{ts_min:.6f}, {ts_max:.6f}] (unit: seconds)")
                except Exception:
                    pass
                self._time_scale_logged = True
        targets = None
        if y_batch is not None and y_batch.shape[1] >= 3:
            h_true = self._denormalize_tensor(y_batch[:, 0:1, :, :], 'height')
            vx_true = self._denormalize_tensor(y_batch[:, 1:2, :, :], 'velocity_x')
            vy_true = self._denormalize_tensor(y_batch[:, 2:3, :, :], 'velocity_y')
            targets = {'h': h_true, 'vx': vx_true, 'vy': vy_true}
        return inputs, targets
    def compute_weighted_sample_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss_fn = nn.SmoothL1Loss(reduction='none')
        if y_pred.dim() == 4:
            h_pred = self._denormalize_tensor(y_pred[:, 0:1, ...], 'height')
            vx_pred = self._denormalize_tensor(y_pred[:, 1:2, ...], 'velocity_x')
            vy_pred = self._denormalize_tensor(y_pred[:, 2:3, ...], 'velocity_y')
            h_true = self._denormalize_tensor(y_true[:, 0:1, ...], 'height')
            vx_true = self._denormalize_tensor(y_true[:, 1:2, ...], 'velocity_x')
            vy_true = self._denormalize_tensor(y_true[:, 2:3, ...], 'velocity_y')
        else:
            h_pred = self._denormalize_tensor(y_pred[:, 0:1], 'height')
            vx_pred = self._denormalize_tensor(y_pred[:, 1:2], 'velocity_x')
            vy_pred = self._denormalize_tensor(y_pred[:, 2:3], 'velocity_y')
            h_true = self._denormalize_tensor(y_true[:, 0:1], 'height')
            vx_true = self._denormalize_tensor(y_true[:, 1:2], 'velocity_x')
            vy_true = self._denormalize_tensor(y_true[:, 2:3], 'velocity_y')
        lh = loss_fn(h_pred, h_true)
        lvx = loss_fn(vx_pred, vx_true)
        lvy = loss_fn(vy_pred, vy_true)
        mask = (h_true > self.active_height_threshold).float()
        aw = self.active_region_weight
        lh = lh * (1.0 + (aw - 1.0) * mask)
        lvx = lvx * (1.0 + (aw - 1.0) * mask)
        lvy = lvy * (1.0 + (aw - 1.0) * mask)
        lh_mean = lh.view(lh.shape[0], -1).mean(dim=1)
        lvx_mean = lvx.view(lvx.shape[0], -1).mean(dim=1)
        lvy_mean = lvy.view(lvy.shape[0], -1).mean(dim=1)
        sample_loss = self.height_loss_w * lh_mean + self.vx_loss_w * lvx_mean + self.vy_loss_w * lvy_mean
        if sample_weights is not None:
            weighted_loss = (sample_loss * sample_weights).mean()
        else:
            weighted_loss = sample_loss.mean()
        return self._check_numerical_stability(weighted_loss, "Data Loss")
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        weights = self.weight_scheduler.get_weights(epoch)
        data_only_epochs = int(getattr(self.weight_scheduler, 'data_only_epochs', 20))
        physics_intro_epochs = int(getattr(self.weight_scheduler, 'physics_intro_epochs', 50))
        if epoch < data_only_epochs:
            active_sampling_rate = 0.05
        elif epoch < data_only_epochs + physics_intro_epochs:
            active_sampling_rate = 0.10
        else:
            active_sampling_rate = 0.6
        current_batch_size = self.batch_scheduler.get_batch_size(epoch)
        if hasattr(self, '_last_batch_size') and self._last_batch_size != current_batch_size:
            pass
        self._last_batch_size = current_batch_size
        epoch_losses_tensor = {
            : torch.tensor(0.0, device=self.device),
            : torch.tensor(0.0, device=self.device),
            : torch.tensor(0.0, device=self.device),
            : torch.tensor(0.0, device=self.device),
            : torch.tensor(0.0, device=self.device)
        }
        epoch_losses = {
            : self.scheduler.get_last_lr()[0]
        }
        loss_clipping_count = 0
        max_clipping_warnings = 1  
        num_batches = 0
        for batch_idx, batch_data in enumerate(self.train_loader):
            if isinstance(batch_data, dict):
                X_batch = batch_data['input'].to(self.device)
                y_batch = batch_data['target'].to(self.device)
                sample_weights = None
                num_time_steps = batch_data.get('num_time_steps', None)
                if num_time_steps is not None:
                    if isinstance(num_time_steps, torch.Tensor):
                        num_time_steps_tensor = num_time_steps.to(self.device)
                    else:
                        try:
                            num_time_steps_tensor = torch.tensor(num_time_steps, device=self.device, dtype=torch.float32)
                        except Exception:
                            num_time_steps_tensor = torch.full((X_batch.shape[0],), float(num_time_steps), device=self.device)
                else:
                    num_time_steps_tensor = None
            else:
                if len(batch_data) == 3:
                    X_batch, y_batch, sample_weights = batch_data
                    sample_weights = sample_weights.to(self.device)
                else:
                    X_batch, y_batch = batch_data
                    sample_weights = None
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                num_time_steps_tensor = None
            self.optimizer.zero_grad()
            B, C_in, H, W = X_batch.shape
            dummy_pred = {
                : torch.zeros(B, 1, H, W, device=self.device),
                : torch.zeros(B, 1, H, W, device=self.device),
                : torch.zeros(B, 1, H, W, device=self.device),
            }
            inputs, targets = self._assemble_physics_inputs(
                X_batch=X_batch,
                predictions_phys=dummy_pred,
                y_batch=y_batch,
                include_time_and_coords=True,
                sample_meta={'num_time_steps': num_time_steps_tensor} if num_time_steps_tensor is not None else None
            )
            if batch_idx == 0:
                logging.info("Batch 0: Physics input assembly completed")
            h_true_phys = targets['h'] if targets is not None else None
            active_threshold = float(getattr(self.physics_loss, 'active_height_threshold', 0.05))
            if h_true_phys is None:
                active_mask = torch.zeros(B, 1, H, W, dtype=torch.bool, device=self.device)
            else:
                active_mask = (h_true_phys > active_threshold)
            inactive_mask = ~active_mask
            p_active_attr = getattr(self.physics_loss, 'pde_active_pixel_ratio', None)
            if p_active_attr is not None:
                p_active = max(0.0, min(1.0, float(p_active_attr)))
                sample_active = (torch.rand(B, 1, H, W, device=self.device) < p_active) & active_mask
            else:
                sample_active = active_mask
            p_inactive_attr = getattr(self.physics_loss, 'pde_inactive_pixel_ratio', None)
            if p_inactive_attr is not None:
                p_inactive = max(0.0, min(1.0, float(p_inactive_attr)))
                sampled_inactive = inactive_mask & (torch.rand(B, 1, H, W, device=self.device) < p_inactive)
            else:
                sampled_inactive = torch.zeros_like(active_mask, dtype=torch.bool)
            if epoch < data_only_epochs:
                boundary_ratio = 0.2
            elif epoch < data_only_epochs + physics_intro_epochs:
                boundary_ratio = 0.3
            else:
                boundary_ratio = 0.5
            boundary_mask_full = inputs['boundary_mask'] > 0.5
            sample_boundary = (torch.rand(B, 1, H, W, device=self.device) < boundary_ratio) & boundary_mask_full
            selected_mask = sample_active | sampled_inactive | sample_boundary
            if batch_idx % 50 == 0:
                selected_count = int(selected_mask.sum().item())
                active_count = int(active_mask.sum().item())
                inactive_count = int((~active_mask).sum().item())
                logging.debug(
                )
            sel_idx = torch.nonzero(selected_mask, as_tuple=False)
            if sel_idx.shape[0] == 0:
                data_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                physics_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                boundary_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                initial_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                b_idx = sel_idx[:, 0]
                y_idx = sel_idx[:, 2]
                x_idx = sel_idx[:, 3]
                centers_coords = torch.stack([b_idx, y_idx, x_idx], dim=1)  
                valid_left = x_idx > 0
                left_coords = torch.stack([b_idx[valid_left], y_idx[valid_left], x_idx[valid_left] - 1], dim=1) if torch.any(valid_left) else torch.empty(0, 3, dtype=torch.long, device=self.device)
                valid_right = x_idx < (W - 1)
                right_coords = torch.stack([b_idx[valid_right], y_idx[valid_right], x_idx[valid_right] + 1], dim=1) if torch.any(valid_right) else torch.empty(0, 3, dtype=torch.long, device=self.device)
                valid_up = y_idx > 0
                up_coords = torch.stack([b_idx[valid_up], y_idx[valid_up] - 1, x_idx[valid_up]], dim=1) if torch.any(valid_up) else torch.empty(0, 3, dtype=torch.long, device=self.device)
                valid_down = y_idx < (H - 1)
                down_coords = torch.stack([b_idx[valid_down], y_idx[valid_down] + 1, x_idx[valid_down]], dim=1) if torch.any(valid_down) else torch.empty(0, 3, dtype=torch.long, device=self.device)
                all_required_coords = torch.cat([centers_coords, left_coords, right_coords, up_coords, down_coords], dim=0)
                unique_coords, inverse = torch.unique(all_required_coords, dim=0, return_inverse=True)
                n_centers = centers_coords.shape[0]
                offset = 0
                centers_inv = inverse[offset:offset + n_centers]; offset += n_centers
                n_left = left_coords.shape[0]
                left_inv = inverse[offset:offset + n_left]; offset += n_left
                n_right = right_coords.shape[0]
                right_inv = inverse[offset:offset + n_right]; offset += n_right
                n_up = up_coords.shape[0]
                up_inv = inverse[offset:offset + n_up]; offset += n_up
                n_down = down_coords.shape[0]
                down_inv = inverse[offset:offset + n_down]
                if self.gradient_method == 'autograd':
                    gradient_results = self._compute_autograd_spatial_gradients(
                        X_batch, centers_coords, self.gradient_sample_size
                    )
                    sampled_centers_coords = gradient_results['sampled_indices']
                    h_center = gradient_results['h']
                    vx_center = gradient_results['vx']
                    vy_center = gradient_results['vy']
                    dh_dx = gradient_results['dh_dx']
                    dh_dy = gradient_results['dh_dy']
                    dvx_dx = gradient_results['dvx_dx']
                    dvx_dy = gradient_results['dvx_dy']
                    dvy_dx = gradient_results['dvy_dx']
                    dvy_dy = gradient_results['dvy_dy']
                    x_points_center = X_batch[sampled_centers_coords[:, 0], :, sampled_centers_coords[:, 1], sampled_centers_coords[:, 2]]
                    x_points_center.requires_grad_(True)
                    y_pred_center_norm = self.model.forward_points(x_points_center)
                    y_true_points_norm = y_batch[sampled_centers_coords[:, 0], :, sampled_centers_coords[:, 1], sampled_centers_coords[:, 2]]
                    data_loss = self.compute_weighted_sample_loss(y_pred_center_norm, y_true_points_norm, None)
                    center_b_idx = sampled_centers_coords[:, 0]
                    center_y_idx = sampled_centers_coords[:, 1]
                    center_x_idx = sampled_centers_coords[:, 2]
                    boundary_mask_pts = inputs['boundary_mask'][center_b_idx, 0, center_y_idx, center_x_idx].unsqueeze(1)
                    initial_mask_pts = inputs['initial_mask'][center_b_idx, 0, center_y_idx, center_x_idx].unsqueeze(1)
                else:
                    if batch_idx == 0:
                        logging.info("Batch 0: Entering finite difference branch")
                    try:
                        x_points_all = X_batch[unique_coords[:, 0], :, unique_coords[:, 1], unique_coords[:, 2]]
                        if batch_idx == 0: logging.info("Batch 0: x_points_all extraction completed")
                    except Exception as e:
                        logging.error(f"Extraction of x_points_all failed: {e}")
                        raise e
                    with torch.no_grad():
                        y_pred_all_norm = self.model.forward_points(x_points_all)
                        if batch_idx == 0: logging.info("Batch 0: forward_points completed")
                    x_points_center = X_batch[centers_coords[:, 0], :, centers_coords[:, 1], centers_coords[:, 2]]
                    x_points_center.requires_grad_(True)
                    y_pred_center_norm = self.model.forward_points(x_points_center)
                    if batch_idx == 0: logging.info("Batch 0: center forward completed")
                    y_true_points_norm = y_batch[centers_coords[:, 0], :, centers_coords[:, 1], centers_coords[:, 2]]
                    data_loss = self.compute_weighted_sample_loss(y_pred_center_norm, y_true_points_norm, None)
                    h_center = self._denormalize_tensor(y_pred_center_norm[:, 0:1], 'height')
                    vx_center = self._denormalize_tensor(y_pred_center_norm[:, 1:2], 'velocity_x')
                    vy_center = self._denormalize_tensor(y_pred_center_norm[:, 2:3], 'velocity_y')
                    h_all = self._denormalize_tensor(y_pred_all_norm[:, 0:1], 'height')
                    vx_all = self._denormalize_tensor(y_pred_all_norm[:, 1:2], 'velocity_x')
                    vy_all = self._denormalize_tensor(y_pred_all_norm[:, 2:3], 'velocity_y')
                    if batch_idx == 0: logging.info("Batch 0: Denormalization completed")
                    h_left = h_center.clone(); h_right = h_center.clone(); h_up = h_center.clone(); h_down = h_center.clone()
                    vx_left = vx_center.clone(); vx_right = vx_center.clone(); vx_up = vx_center.clone(); vx_down = vx_center.clone()
                    vy_left = vy_center.clone(); vy_right = vy_center.clone(); vy_up = vy_center.clone(); vy_down = vy_center.clone()
                    if n_left > 0:
                        h_left[valid_left] = h_all[left_inv]; vx_left[valid_left] = vx_all[left_inv]; vy_left[valid_left] = vy_all[left_inv]
                    if n_right > 0:
                        h_right[valid_right] = h_all[right_inv]; vx_right[valid_right] = vx_all[right_inv]; vy_right[valid_right] = vy_all[right_inv]
                    if n_up > 0:
                        h_up[valid_up] = h_all[up_inv]; vx_up[valid_up] = vx_all[up_inv]; vy_up[valid_up] = vy_all[up_inv]
                    if n_down > 0:
                        h_down[valid_down] = h_all[down_inv]; vx_down[valid_down] = vx_all[down_inv]; vy_down[valid_down] = vy_all[down_inv]
                    if batch_idx == 0: logging.info("Batch 0: Neighbors assignment completed")
                    try:
                        dx_phys = float(getattr(self.physics_loss, 'dx', 1.0))
                        dy_phys = float(getattr(self.physics_loss, 'dy', 1.0))
                        both_x = valid_left & valid_right
                        only_right = ~valid_left & valid_right
                        only_left = valid_left & ~valid_right
                        dh_dx = torch.zeros_like(h_center); dvx_dx = torch.zeros_like(vx_center); dvy_dx = torch.zeros_like(vy_center)
                        dh_dx[both_x] = (h_right[both_x] - h_left[both_x]) / (2.0 * dx_phys)
                        dvx_dx[both_x] = (vx_right[both_x] - vx_left[both_x]) / (2.0 * dx_phys)
                        dvy_dx[both_x] = (vy_right[both_x] - vy_left[both_x]) / (2.0 * dx_phys)
                        dh_dx[only_right] = (h_right[only_right] - h_center[only_right]) / dx_phys
                        dvx_dx[only_right] = (vx_right[only_right] - vx_center[only_right]) / dx_phys
                        dvy_dx[only_right] = (vy_right[only_right] - vy_center[only_right]) / dx_phys
                        dh_dx[only_left] = (h_center[only_left] - h_left[only_left]) / dx_phys
                        dvx_dx[only_left] = (vx_center[only_left] - vx_left[only_left]) / dx_phys
                        dvy_dx[only_left] = (vy_center[only_left] - vy_left[only_left]) / dx_phys
                        if batch_idx == 0: logging.info("Batch 0: x-derivative calculation completed")
                        both_y = valid_up & valid_down
                        only_down = ~valid_up & valid_down
                        only_up = valid_up & ~valid_down
                        dh_dy = torch.zeros_like(h_center); dvx_dy = torch.zeros_like(vx_center); dvy_dy = torch.zeros_like(vy_center)
                        dh_dy[both_y] = (h_down[both_y] - h_up[both_y]) / (2.0 * dy_phys)
                        dvx_dy[both_y] = (vx_down[both_y] - vx_up[both_y]) / (2.0 * dy_phys)
                        dvy_dy[both_y] = (vy_down[both_y] - vy_up[both_y]) / (2.0 * dy_phys)
                        dh_dy[only_down] = (h_down[only_down] - h_center[only_down]) / dy_phys
                        dvx_dy[only_down] = (vx_down[only_down] - vx_center[only_down]) / dy_phys
                        dvy_dy[only_down] = (vy_down[only_down] - vy_center[only_down]) / dy_phys
                        dh_dy[only_up] = (h_center[only_up] - h_up[only_up]) / dy_phys
                        dvx_dy[only_up] = (vx_center[only_up] - vx_up[only_up]) / dy_phys
                        dvy_dy[only_up] = (vy_center[only_up] - vy_up[only_up]) / dy_phys
                        if batch_idx == 0: logging.info("Batch 0: y-derivative calculation completed")
                    except Exception as e:
                        logging.error(f"Spatial derivative calculation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        raise e
                if batch_idx == 0:
                    logging.info("Batch 0: Starting time derivative calculation")
                center_b_idx = centers_coords[:, 0]
                center_y_idx = centers_coords[:, 1]
                center_x_idx = centers_coords[:, 2]
                dt_phys = float(getattr(self.physics_loss, 'dt', 1.0))
                h_input_norm = X_batch[center_b_idx, 0:1, center_y_idx, center_x_idx]
                vx_input_norm = X_batch[center_b_idx, 1:2, center_y_idx, center_x_idx]
                vy_input_norm = X_batch[center_b_idx, 2:3, center_y_idx, center_x_idx]
                h_input_phys = self._denormalize_tensor(h_input_norm, 'height')
                vx_input_phys = self._denormalize_tensor(vx_input_norm, 'velocity_x')
                vy_input_phys = self._denormalize_tensor(vy_input_norm, 'velocity_y')
                dh_dt = (h_center - h_input_phys) / dt_phys
                dvx_dt = (vx_center - vx_input_phys) / dt_phys
                dvy_dt = (vy_center - vy_input_phys) / dt_phys
                dem_center = inputs['dem'][center_b_idx, 0, center_y_idx, center_x_idx].unsqueeze(1)
                dzdx_center = inputs['dz_dx'][center_b_idx, 0, center_y_idx, center_x_idx].unsqueeze(1)
                dzdy_center = inputs['dz_dy'][center_b_idx, 0, center_y_idx, center_x_idx].unsqueeze(1)
                mu_center = self._denormalize_tensor(X_batch[center_b_idx, 10:11, center_y_idx, center_x_idx], 'mu_0')
                xi_center = self._denormalize_tensor(X_batch[center_b_idx, 11:12, center_y_idx, center_x_idx], 'xi_0')
                rho_center = self._denormalize_tensor(X_batch[center_b_idx, 12:13, center_y_idx, center_x_idx], 'rho')
                if weights['physics'] > 0:
                    boundary_mask_pts = inputs['boundary_mask'][center_b_idx, 0, center_y_idx, center_x_idx].unsqueeze(1)  
                    initial_mask_pts = inputs['initial_mask'][center_b_idx, 0, center_y_idx, center_x_idx].unsqueeze(1)    
                    release_mask_center = X_batch[center_b_idx, 9, center_y_idx, center_x_idx].unsqueeze(1)  
                    physics_loss = self.physics_loss.compute_residual_loss_on_points(
                        states={'h': h_center, 'vx': vx_center, 'vy': vy_center},
                        grads={
                            : dh_dt, 'dh_dx': dh_dx, 'dh_dy': dh_dy,
                            : dvx_dt, 'dvx_dx': dvx_dx, 'dvx_dy': dvx_dy,
                            : dvy_dt, 'dvy_dx': dvy_dx, 'dvy_dy': dvy_dy
                        },
                        static={'dzdx': dzdx_center, 'dzdy': dzdy_center, 'mu': mu_center, 'xi': xi_center, 'rho': rho_center,
                                : boundary_mask_pts, 'initial_mask': initial_mask_pts, 'release_mask': release_mask_center},
                        active_mask=None,
                        sample_ratio=active_sampling_rate
                    )
                    vel_mag_center = torch.sqrt(vx_center**2 + vy_center**2 + 1e-8)
                    inflow_thr = 1.0 if epoch < (data_only_epochs + physics_intro_epochs) else 0.5
                    excessive_v = torch.relu(vel_mag_center - inflow_thr)
                    boundary_loss = torch.mean(excessive_v**2 * boundary_mask_pts)
                    init_vel_excess = torch.relu(vel_mag_center - float(getattr(self.physics_loss, 'initial_velocity_threshold', 0.5)))
                    init_vel_loss = torch.mean(init_vel_excess**2 * initial_mask_pts)
                    min_rel = float(getattr(self.physics_loss, 'release_height_min', 0.5))
                    max_rel = float(getattr(self.physics_loss, 'release_height_max', 8.0))
                    non_rel_thr = float(getattr(self.physics_loss, 'non_release_height_threshold', 0.1))
                    height_max_lim = float(getattr(self.physics_loss, 'height_max_limit', 50.0))
                    is_release_pts = (release_mask_center > 0.5).float() * initial_mask_pts
                    is_non_release_pts = (release_mask_center <= 0.5).float() * initial_mask_pts
                    release_h_too_low = torch.relu(min_rel - h_center)
                    release_h_too_high = torch.relu(h_center - max_rel)
                    release_h_loss = torch.mean((release_h_too_low**2 + release_h_too_high**2) * is_release_pts)
                    non_release_h_excess = torch.relu(h_center - non_rel_thr)
                    non_release_h_loss = torch.mean(non_release_h_excess**2 * is_non_release_pts)
                    negative_h_penalty = torch.mean(torch.relu(-h_center)**2 * initial_mask_pts)
                    excessive_h_penalty = torch.mean(torch.relu(h_center - height_max_lim)**2 * initial_mask_pts)
                    initial_loss = (1.0*init_vel_loss + 0.5*release_h_loss + 0.3*non_release_h_loss + 0.4*negative_h_penalty + 0.3*excessive_h_penalty)
                    b_w = float(getattr(self.physics_loss, 'boundary_constraint_weight', 0.3))
                    i_w = float(getattr(self.physics_loss, 'initial_constraint_weight', 0.1))
                    comps = getattr(self.physics_loss, 'last_component_losses', None)
                    if comps:
                        def clamp(v):
                            return max(0.2, min(1.0, v))
                        total_comp = (comps.get('continuity', 0.0) + comps.get('momentum_x', 0.0) + comps.get('momentum_y', 0.0) + comps.get('height', 0.0) + comps.get('boundary', 0.0) + comps.get('initial', 0.0) + 1e-8)
                        b_share = comps.get('boundary', 0.0) / total_comp
                        i_share = comps.get('initial', 0.0) / total_comp
                        b_w = clamp(b_w * (0.8 if b_share > 0.4 else 1.1))
                        i_w = clamp(i_w * (0.8 if i_share > 0.4 else 1.1))
                        try:
                            self.physics_loss.boundary_constraint_weight = b_w
                            self.physics_loss.initial_constraint_weight = i_w
                        except Exception:
                            pass
                    physics_loss = physics_loss + b_w * boundary_loss + i_w * initial_loss
                else:
                    physics_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    physics_loss = self._check_numerical_stability(physics_loss, "Physics Loss (point-level)")
            total_loss = (weights['data'] * data_loss +
                         weights['physics'] * physics_loss)
            total_loss = self._check_numerical_stability(total_loss, "Total Loss")
            if total_loss.item() > self.max_loss_value:
                loss_clipping_count += 1
                total_loss = torch.clamp(total_loss, max=self.max_loss_value)
            total_loss.backward()
            if self.verify_gradients and batch_idx % 100 == 0:  
                gradient_info = self._verify_gradient_flow(total_loss)
                if not gradient_info['gradient_flow_ok']:
                    logging.warning(
                    )
                elif batch_idx % 500 == 0:  
                    logging.info(
                    )
            if not self._check_gradients():
                logging.warning(f"Gradients contain NaN or Inf, skipping this batch update")
                self.optimizer.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
            self.optimizer.step()
            epoch_losses_tensor['data_loss'] += data_loss.detach()
            epoch_losses_tensor['physics_loss'] += physics_loss.detach()
            epoch_losses_tensor['boundary_loss'] += boundary_loss.detach()
            epoch_losses_tensor['initial_loss'] += initial_loss.detach()
            epoch_losses_tensor['total_loss'] += total_loss.detach()
            num_batches += 1
        for key, tensor_loss in epoch_losses_tensor.items():
            epoch_losses[key] = (tensor_loss / max(num_batches, 1)).item()
        if loss_clipping_count > num_batches * 0.1:
            logging.warning(f"Frequent loss clipping in this epoch: {loss_clipping_count}/{num_batches} batch ({loss_clipping_count/num_batches*100:.1f}%)")
        return epoch_losses
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_losses = {
            : 0.0,
            : 0.0,
            : 0.0
        }
        num_batches = 0
        def _init_range():
            return {'min': float('inf'), 'max': float('-inf')}
        pred_ranges = {
            : _init_range(),
            : _init_range(),
            : _init_range()
        }
        true_ranges = {
            : _init_range(),
            : _init_range(),
            : _init_range()
        }
        with torch.no_grad():
            for batch_data in self.val_loader:
                if isinstance(batch_data, dict):
                    X_batch = batch_data['input'].to(self.device)
                    y_batch = batch_data['target'].to(self.device)
                else:
                    if len(batch_data) == 3:
                        X_batch, y_batch, _ = batch_data
                    else:
                        X_batch, y_batch = batch_data
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)
                data_loss = nn.MSELoss()(y_pred, y_batch)
                if y_pred.shape[1] >= 3:
                    predictions_norm = {
                        : y_pred[:, 0:1, :, :],
                        : y_pred[:, 1:2, :, :],
                        : y_pred[:, 2:3, :, :]
                    }
                else:
                    predictions_norm = {
                        : y_pred,
                        : torch.zeros_like(y_pred),
                        : torch.zeros_like(y_pred)
                    }
                predictions = {
                    : self._denormalize_tensor(predictions_norm['h'], 'height'),
                    : self._denormalize_tensor(predictions_norm['vx'], 'velocity_x'),
                    : self._denormalize_tensor(predictions_norm['vy'], 'velocity_y')
                }
                if y_batch is not None:
                    if y_batch.dim() == 4 and y_batch.shape[1] >= 3:
                        targets_denorm = {
                            : self._denormalize_tensor(y_batch[:, 0:1, :, :], 'height'),
                            : self._denormalize_tensor(y_batch[:, 1:2, :, :], 'velocity_x'),
                            : self._denormalize_tensor(y_batch[:, 2:3, :, :], 'velocity_y')
                        }
                    else:
                        targets_denorm = {
                            : self._denormalize_tensor(y_batch[:, 0:1, :, :] if y_batch.dim() == 4 else y_batch, 'height'),
                            : torch.zeros_like(predictions['vx']),
                            : torch.zeros_like(predictions['vy'])
                        }
                else:
                    targets_denorm = {
                        : torch.zeros_like(predictions['h']),
                        : torch.zeros_like(predictions['vx']),
                        : torch.zeros_like(predictions['vy'])
                    }
                def _update_range(rng, tensor):
                    try:
                        tmin = float(tensor.min().item())
                        tmax = float(tensor.max().item())
                        rng['min'] = min(rng['min'], tmin)
                        rng['max'] = max(rng['max'], tmax)
                    except Exception:
                        pass
                _update_range(pred_ranges['h'], predictions['h'])
                _update_range(pred_ranges['vx'], predictions['vx'])
                _update_range(pred_ranges['vy'], predictions['vy'])
                _update_range(true_ranges['h'], targets_denorm['h'])
                _update_range(true_ranges['vx'], targets_denorm['vx'])
                _update_range(true_ranges['vy'], targets_denorm['vy'])
                inputs, targets = self._assemble_physics_inputs(
                    X_batch=X_batch,
                    predictions_phys=predictions,
                    y_batch=y_batch,
                    include_time_and_coords=False  
                )
                try:
                    with torch.enable_grad():
                        physics_loss = self.physics_loss.compute_dimensionless_physics_loss(
                            x_batch=X_batch,
                            predictions=None,
                            targets=None,
                            model=self.model,
                            active_sampling_rate=self.physics_loss.default_active_sampling_rate,
                            static_data=None,
                            physics_params=None
                        )
                except Exception:
                    physics_loss = torch.tensor(0.0, device=self.device)
                val_losses['val_data_loss'] += data_loss.item()
                val_losses['val_physics_loss'] += physics_loss.item()
                comps = getattr(self.physics_loss, 'last_component_losses', None)
                if comps and ((self._current_epoch + 1) % 20 == 0 or self._current_epoch == 0):
                    logging.info(f"Validate physics components: cont={comps.get('continuity',0):.6f}, mx={comps.get('momentum_x',0):.6f}, my={comps.get('momentum_y',0):.6f}, "
                                 )
                val_losses['val_loss'] += (data_loss + physics_loss).item()
                num_batches += 1
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)
        current_epoch = getattr(self, '_current_epoch', 0)
        if (current_epoch + 1) % 20 == 0 or current_epoch == 0:
            def _fmt(rng):
                return f"[{rng['min']:.4f}, {rng['max']:.4f}]"
            logging.info("Validation stage prediction/truth range statistics:")
            logging.info(f"  Height H Predicted: {_fmt(pred_ranges['h'])}  |  Truth: {_fmt(true_ranges['h'])}")
            logging.info(f"  Velocity Ux Predicted: {_fmt(pred_ranges['vx'])}  |  Truth: {_fmt(true_ranges['vx'])}")
            logging.info(f"  Velocity Uy Predicted: {_fmt(pred_ranges['vy'])}  |  Truth: {_fmt(true_ranges['vy'])}")
        return val_losses
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
        is_interval_best: bool = False
    ):
        checkpoint = {
            : epoch,
            : self.model.state_dict(),
            : self.optimizer.state_dict(),
            : self.scheduler.state_dict(),
            : val_loss,
            : dict(self.train_history),
            : dict(self.val_history),
            : self.config
        }
        if epoch % 10 == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            logging.info(f"Best model saved,Epoch {epoch + 1}, Val Loss: {val_loss:.6f}")
        if is_interval_best:
            interval_path = self.checkpoint_dir / f'interval_best_{epoch - self.interval_size + 1}_{epoch}.pth'
            torch.save(checkpoint, interval_path)
            logging.info(f"Interval best model saved,Epoch {epoch + 1}, Val Loss: {val_loss:.6f}")
    def train(self, start_epoch: int = 0) -> Dict[str, List[float]]:
        print(f"Starting PINN training, total{self.num_epochs}epochs...")
        logging.info("Starting enhanced PINN training...")
        start_time = time.time()
        for epoch in range(start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            self._current_epoch = epoch
            weights = self.weight_scheduler.get_weights(epoch)
            current_batch_size = self.batch_scheduler.get_batch_size(epoch)
            if hasattr(self, '_last_batch_size') and self._last_batch_size != current_batch_size:
                print(f"Epoch {epoch + 1}: reducing batch size to {current_batch_size}")
                logging.info(f"Epoch {epoch + 1}: reducing batch size to {current_batch_size}")
            if (epoch + 1) % 50 == 0 or epoch == 0:
                logging.info(f"\n{'='*80}")
                logging.info(f"Epoch {epoch + 1}/{self.num_epochs}")
                logging.info(f"Weight config: Data={weights['data']:.3f}, Physics={weights['physics']:.3f}, "
                            )
                logging.info(f"Batch size: {current_batch_size}")
            train_losses = self.train_epoch(epoch)
            try:
                self.weight_scheduler.update_running_losses(train_losses.get('data_loss', None),
                                                           train_losses.get('physics_loss', None))
            except Exception:
                pass
            val_losses = self.validate()
            self.scheduler.step()
            if self.plateau_scheduler is not None:
                try:
                    self.plateau_scheduler.step(val_losses['val_loss'])
                except Exception:
                    pass
            for loss_name, loss_value in train_losses.items():
                self.train_history[loss_name].append(loss_value)
            for loss_name, loss_value in val_losses.items():
                self.val_history[loss_name].append(loss_value)
            is_best = val_losses['val_loss'] < self.best_val_loss
            if is_best:
                improvement = (self.best_val_loss - val_losses['val_loss']) if self.best_val_loss != float('inf') else float('inf')
                self.best_val_loss = val_losses['val_loss']
                self.patience_counter = 0
                print_impr_pct = float(self.config.get('training', {}).get('best_print_min_improvement_pct', 0.05))
                if improvement == float('inf') or (improvement / max(self.best_val_loss + improvement, 1e-8)) >= print_impr_pct or epoch == 0:
                    print(f"New best model saved,Val Loss: {self.best_val_loss:.6f}")
                logging.info(f"Best model saved,Epoch {epoch + 1}, Val Loss: {val_losses['val_loss']:.6f}")
            else:
                self.patience_counter += 1
            is_interval_best = False
            if val_losses['val_loss'] < self.interval_best_val_loss:
                self.interval_best_val_loss = val_losses['val_loss']
                self.interval_best_epoch = epoch
            if (epoch + 1) % self.interval_size == 0:
                if self.interval_best_epoch >= (epoch + 1 - self.interval_size):
                    is_interval_best = True
                    print(f"Interval best model saved,Epoch {self.interval_best_epoch + 1}, Val Loss: {self.interval_best_val_loss:.6f}")
                    logging.info(f"Interval best model saved,Epoch {epoch + 1}, Val Loss: {val_losses['val_loss']:.6f}")
                self.interval_best_val_loss = float('inf')
                self.interval_best_epoch = -1
            if epoch % 10 == 0 or is_best or is_interval_best:
                self.save_checkpoint(epoch, val_losses['val_loss'], is_best, is_interval_best)
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - start_time
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Data Loss: {train_losses['data_loss']:.6f}, Physics Loss: {train_losses['physics_loss']:.6f}, "
                      )
            if (epoch + 1) % 10 == 0 or epoch == 0 or is_best:
                logging.info(f"Epoch {epoch + 1}/{self.num_epochs} - {epoch_time:.2f}s")
                logging.info(f"Data Loss: {train_losses['data_loss']:.6f}, "
                            )
                logging.info(f"Val Loss: {val_losses['val_loss']:.6f}, "
                            )
            else:
                logging.info(f"Epoch {epoch + 1}/{self.num_epochs}: Total Loss={train_losses['total_loss']:.6f}, "
                            )
            if self.patience_counter >= self.patience:
                print(f"Early stopping: {self.patience}epochs without Val Loss improvement")
                logging.info(f"Early stopping triggered at epoch{epoch + 1}")
                break
        print(f"Final model saved,Val Loss: {val_losses['val_loss']:.6f}")
        logging.info("Enhanced PINN training completed!")
        self._save_training_history()
        return dict(self.train_history), dict(self.val_history)
    def _generate_boundary_and_initial_masks(
        self, 
        X_batch: torch.Tensor, 
        output_shape: torch.Size
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = output_shape[0]
        height = output_shape[2] if len(output_shape) > 2 else 1
        width = output_shape[3] if len(output_shape) > 3 else 1
        boundary_mask = torch.zeros(batch_size, 1, height, width, device=self.device)
        if height > 1 and width > 1:
            boundary_mask[:, :, 0, :] = 1.0      
            boundary_mask[:, :, -1, :] = 1.0     
            boundary_mask[:, :, :, 0] = 1.0      
            boundary_mask[:, :, :, -1] = 1.0     
        else:
            num_boundary_points = max(1, batch_size // 4)
            boundary_indices = torch.randperm(batch_size)[:num_boundary_points]
            boundary_mask[boundary_indices] = 1.0
        initial_mask = torch.zeros(batch_size, 1, height, width, device=self.device)
        if X_batch.shape[1] >= 6:
            if len(X_batch.shape) == 4:  
                t_values = X_batch[:, 5, 0, 0]
            else:  
                t_values = X_batch[:, 5]
            t_min = torch.min(t_values)
            t_threshold = t_min + (torch.max(t_values) - t_min) * 0.05  
            is_initial = t_values <= t_threshold
            for i in range(batch_size):
                if is_initial[i]:
                    initial_mask[i, :, :, :] = 1.0
        else:
            num_initial_points = max(1, batch_size // 3)
            initial_indices = torch.randperm(batch_size)[:num_initial_points]
            for idx in initial_indices:
                initial_mask[idx, :, :, :] = 1.0
        return boundary_mask, initial_mask
    def _save_training_history(self):
        history = {
            : dict(self.train_history),
            : dict(self.val_history),
            : self.best_val_loss,
            : self.config
        }
        history_path = self.checkpoint_dir / 'enhanced_training_history.pth'
        torch.save(history, history_path)
        logging.info(f"Training history saved to: {history_path}")
    def plot_training_history(self, save_path: Optional[str] = None):
        try:
            preferred_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Arial Unicode MS']
            available = {f.name for f in fm.fontManager.ttflist}
            for fname in preferred_fonts:
                if fname in available:
                    current_fonts = matplotlib.rcParams.get('font.sans-serif', [])
                    matplotlib.rcParams['font.sans-serif'] = [fname] + list(current_fonts)
                    break
            try:
                from matplotlib_config import setup_matplotlib_for_chinese
                setup_matplotlib_for_chinese(verbose=False)
            except Exception:
                pass
        except Exception as _:
            pass
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        import matplotlib.ticker as mticker
        for ax in [axes[0, 0], axes[0, 1], axes[1, 1]]:
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
        axes[1, 0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
        epochs = range(1, len(self.train_history['total_loss']) + 1)
        axes[0, 0].plot(epochs, self.train_history['total_loss'], label='Train total loss')
        axes[0, 0].plot(epochs, self.val_history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 1].plot(epochs, self.train_history['data_loss'], label='Data Loss')
        axes[0, 1].plot(epochs, self.train_history['physics_loss'], label='Physics Loss')
        axes[0, 1].plot(epochs, self.train_history['boundary_loss'], label='Boundary Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[1, 0].plot(epochs, self.train_history['learning_rate'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning rate')
        axes[1, 0].set_title('Learning rate schedule')
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        axes[1, 1].plot(epochs, self.val_history['val_data_loss'], label='Val Data Loss')
        axes[1, 1].plot(epochs, self.val_history['val_physics_loss'], label='Val Physics Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Val Loss detailed')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Training history plot saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()
def create_trainer(
    model: AvalanchePINN,
    physics_loss: AvalanchePhysicsLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any]
) -> EnhancedAvalanchePINNTrainer:
    return EnhancedAvalanchePINNTrainer(
        model=model,
        physics_loss=physics_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from torch.utils.data import DataLoader, Dataset
    batch_size = 4
    height, width = 32, 32
    num_samples = 100
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height),
        torch.linspace(0, 1, width),
        indexing='ij'
    )
    x_field = xx.unsqueeze(0).unsqueeze(0)  
    y_field = yy.unsqueeze(0).unsqueeze(0)  
    t_field = torch.full((1, 1, height, width), 0.5, dtype=torch.float32)  
    dem_field = torch.zeros(1, 1, height, width)
    dzdx_field = torch.zeros(1, 1, height, width)
    dzdy_field = torch.zeros(1, 1, height, width)
    placeholder_3 = torch.zeros(1, 3, height, width)
    other_channels = torch.zeros(1, 14 - 9, height, width)
    inputs_4d = torch.cat([
        placeholder_3,      
        x_field,            
        y_field,            
        t_field,            
        dem_field,          
        dzdx_field,         
        dzdy_field,         
        other_channels      
    ], dim=1)
    h_target = 0.1 * torch.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.1)
    h_target = h_target.unsqueeze(0).unsqueeze(0)  
    vx_target = 0.5 * torch.ones(1, 1, height, width)
    vy_target = -0.3 * torch.ones(1, 1, height, width)
    targets_4d = torch.cat([h_target, vx_target, vy_target], dim=1)  
    class SimpleGridDataset(Dataset):
        def __init__(self, inp4d: torch.Tensor, tgt4d: torch.Tensor, num_time_steps: int = 10):
            self.inp4d = inp4d
            self.tgt4d = tgt4d
            self.num_time_steps = num_time_steps
            self.norm_stats = {}
        def __len__(self):
            return num_samples
        def __getitem__(self, idx):
            inp = self.inp4d[0].clone() + 0.01 * torch.randn_like(self.inp4d[0])
            tgt = self.tgt4d[0].clone() + 0.01 * torch.randn_like(self.tgt4d[0])
            return {
                : inp,
                : tgt,
                : self.num_time_steps
            }
    dataset = SimpleGridDataset(inputs_4d, targets_4d, num_time_steps=10)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = AvalanchePINN()
    physics_loss = AvalanchePhysicsLoss()
    config = {
        : {
            : 2,  
            : 5e-4,  
            : 1e-5,  
            : 1.0,  
            : 1,  
            : 2,  
            : 0.3,  
            : 1.2,  
            : 0.2,  
            : 0.5,  
            : batch_size,  
            : batch_size,  
            : 1.0,  
            : 10,  
            : 10  
        },
        : './test_enhanced_checkpoints',
        : './test_enhanced_logs'
    }
    trainer = create_trainer(model, physics_loss, train_loader, val_loader, config)
    print("Starting enhanced PINN trainer test...")
    train_history, val_history = trainer.train()
    print("Enhanced trainer test completed!")
    print(f"Train Epoch: {len(train_history.get('total_loss', []))}")
    print(f"Validation epochs: {len(val_history.get('val_loss', []))}")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    trainer.plot_training_history(save_path='./test_enhanced_training_history.png')
