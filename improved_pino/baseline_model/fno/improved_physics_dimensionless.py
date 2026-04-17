
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
class DimensionlessPhysicsLoss(nn.Module):
    def __init__(self, config: Dict[str, Any], global_data_config, device='cuda'):
        super().__init__()
        self.config = config
        self.global_data_config = global_data_config
        self.device = device
        self.g = config.get('physics', {}).get('gravity', 9.81)  
        self.dx = config.get('data', {}).get('dx', 10.0)  
        self.dy = config.get('data', {}).get('dy', 10.0)  
        self.dt = config.get('data', {}).get('dt', 1.0)   
        physics_config = config.get('physics', {})
        self.compute_in_original_units = physics_config.get('compute_in_original_units', True)
        self.enable_initial_condition_loss = physics_config.get('enable_initial_condition_loss', True)
        self.enable_boundary_condition_loss = physics_config.get('enable_boundary_condition_loss', True)
        self.use_dimensionless = physics_config.get('use_dimensionless', True)
        self._initialize_characteristic_scales()
        self.eps = 1e-8
        self.loss_clamp_min = 1e-8
        self.loss_clamp_max = 1e6
        physics_config = config.get('physics', {})
        self.enable_active_region_sampling = physics_config.get('enable_active_region_sampling', True)
        self.active_height_threshold = physics_config.get('active_height_threshold', 0.05)
        self.active_region_sample_ratio = physics_config.get('active_region_sample_ratio', 0.8)
        self.inactive_region_sample_ratio = physics_config.get('inactive_region_sample_ratio', 0.2)
        self.min_active_points = physics_config.get('min_active_points', 30)
        self.continuity_weight = 1.0
        self.momentum_weight = 1.0
        self.height_constraint_weight = 1.0
        self.eq1_weight = self.continuity_weight  
        self.eq2_weight = self.momentum_weight    
        self.eq3_weight = self.momentum_weight    
        self.zero_grad_count = 0
        logging.info(f"Dimensionless physics loss initialized:")
        logging.info(f"  Spatial scale: dx={self.dx}m, dy={self.dy}m")
        logging.info(f"  Time scale: dt={self.dt}s")
        logging.info(f"  Characteristic height scale: H_char={self.H_char:.3f}m")
        logging.info(f"  Characteristic velocity scale: U_char={self.U_char:.3f}m/s")
        logging.info(f"  Characteristic acceleration scale: A_char={self.A_char:.3f}m/s2")
        logging.info(f"  Dimensionless calculation: {'enabled' if self.use_dimensionless else 'disabled'}")
        logging.info(f"  Active region sampling: {'enabled' if self.enable_active_region_sampling else 'disabled'}")
        if self.enable_active_region_sampling:
            logging.info(f"    Active height threshold: {self.active_height_threshold}m")
            logging.info(f"    Active region sampling ratio: {self.active_region_sample_ratio}")
            logging.info(f"    Inactive region sampling ratio: {self.inactive_region_sample_ratio}")
            logging.info(f"    Minimum active points: {self.min_active_points}")
    def forward(self, predictions: torch.Tensor, x_batch: torch.Tensor, 
                targets: Optional[torch.Tensor] = None, 
                pred_norm: Optional[torch.Tensor] = None,
                model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        if targets is None:
            targets = torch.zeros_like(predictions)
        return self.compute_dimensionless_physics_loss(
            x_batch=x_batch,
            predictions=predictions, 
            targets=targets,
            pred_norm=pred_norm,
            model=model
        )
    def _initialize_characteristic_scales(self):
        try:
            self.H_mean, self.H_std = self.global_data_config.get_height_denorm_params()
            velocity_params = self.global_data_config.get_velocity_denorm_params()
            self.Vx_mean = velocity_params['velocity_x_mean']
            self.Vx_std = velocity_params['velocity_x_std']
            self.Vy_mean = velocity_params['velocity_y_mean']
            self.Vy_std = velocity_params['velocity_y_std']
            self.U_mean = np.sqrt(self.Vx_mean**2 + self.Vy_mean**2)
            self.U_std = np.sqrt(self.Vx_std**2 + self.Vy_std**2)
            self.H_char = max(abs(self.H_mean), self.H_std, 0.1)  
            self.U_char = max(self.U_mean, self.U_std, np.sqrt(self.g * self.H_char), 1.0)  
            self.A_char = self.g  
            self.Froude = self.U_char / np.sqrt(self.g * self.H_char)  
            logging.info(f"Characteristic scale parameters:")
            logging.info(f"  H_mean={self.H_mean:.3f}m, H_std={self.H_std:.3f}m")
            logging.info(f"  U_mean={self.U_mean:.3f}m/s, U_std={self.U_std:.3f}m/s")
            logging.info(f"  Froude number={self.Froude:.3f}")
        except Exception as e:
            logging.warning(f"Failed to get statistical parameters, using defaults: {e}")
            self.H_char = 1.0
            self.U_char = 5.0
            self.A_char = self.g
            self.Froude = 1.0
    def compute_dimensionless_physics_loss(self, 
                                         x_batch: torch.Tensor,
                                         predictions: torch.Tensor,
                                         targets: torch.Tensor,
                                         pred_norm: Optional[torch.Tensor] = None,
                                         model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        try:
            if model is not None:
                x_for_physics = self._prepare_gradient_inputs(x_batch)
                predictions_with_grad = model(x_for_physics)
                physics_data = self._prepare_physics_data(x_for_physics, predictions_with_grad, pred_norm)
            else:
                x_for_grad = self._prepare_gradient_inputs(x_batch)
                physics_data = self._prepare_physics_data(x_for_grad, predictions, pred_norm)
            sampling_mask = self._compute_active_region_sampling_mask(physics_data)
            gradients = self._compute_dimensionless_gradients_sampled(physics_data, sampling_mask)
            pde_residuals = self._compute_dimensionless_pde_residuals_sampled(physics_data, gradients, sampling_mask)
            physics_loss = self._compute_physics_constraints_sampled(pde_residuals, physics_data, sampling_mask)
            return physics_loss
        except Exception as e:
            logging.error(f"Dimensionless physics loss calculation failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    def _prepare_physics_data(self, x_batch: torch.Tensor, predictions: torch.Tensor, 
                            pred_norm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = x_batch.shape[0]
        x_for_grad = x_batch[:, 3:4]  
        y_for_grad = x_batch[:, 4:5]  
        t_for_grad = x_batch[:, 5:6]  
        if not x_for_grad.requires_grad:
            x_for_grad.requires_grad_(True)
        if not y_for_grad.requires_grad:
            y_for_grad.requires_grad_(True)
        if not t_for_grad.requires_grad:
            t_for_grad.requires_grad_(True)
        coord_scale = self.global_data_config.get_coord_scale()
        time_scale = self.global_data_config.get_time_scale()
        x_phys = x_for_grad * coord_scale  
        y_phys = y_for_grad * coord_scale  
        t_phys = t_for_grad * time_scale  
        h_mean, h_std = self.global_data_config.get_height_denorm_params()
        velocity_params = self.global_data_config.get_velocity_denorm_params()
        h_phys = predictions[:, 0:1] * h_std + h_mean  
        vx_phys = predictions[:, 1:2] * velocity_params['velocity_x_std'] + velocity_params['velocity_x_mean']  
        vy_phys = predictions[:, 2:3] * velocity_params['velocity_y_std'] + velocity_params['velocity_y_mean']  
        dem = x_batch[:, 6:7]  
        dzdx_norm = x_batch[:, 7:8]  
        dzdy_norm = x_batch[:, 8:9]  
        gradient_denorm_params = self.global_data_config.get_gradient_denorm_params()
        dzdx_phys = dzdx_norm * gradient_denorm_params['dzdx_std'] + gradient_denorm_params['dzdx_mean']
        dzdy_phys = dzdy_norm * gradient_denorm_params['dzdy_std'] + gradient_denorm_params['dzdy_mean']
        if x_batch.shape[1] != 14:
            raise ValueError(f"Expected 14-channel input, but got{x_batch.shape[1]}channels. Ensure input contains all physical parameters.")
        mu_norm = x_batch[:, 10:11]      
        xi_norm = x_batch[:, 11:12]      
        rho_norm = x_batch[:, 12:13]     
        cohesion_norm = x_batch[:, 13:14] 
        physics_denorm_params = self.global_data_config.get_physics_denorm_params()
        mu_phys = mu_norm * (physics_denorm_params['mu_0_max'] - physics_denorm_params['mu_0_min']) + physics_denorm_params['mu_0_min']
        xi_phys = xi_norm * (physics_denorm_params['xi_0_max'] - physics_denorm_params['xi_0_min']) + physics_denorm_params['xi_0_min']
        rho_phys = rho_norm * (physics_denorm_params['rho_max'] - physics_denorm_params['rho_min']) + physics_denorm_params['rho_min']
        cohesion_phys = cohesion_norm * (physics_denorm_params['cohesion_max'] - physics_denorm_params['cohesion_min']) + physics_denorm_params['cohesion_min']
        mu_phys = torch.clamp(mu_phys, min=physics_denorm_params['mu_0_min'], max=physics_denorm_params['mu_0_max'])
        xi_phys = torch.clamp(xi_phys, min=physics_denorm_params['xi_0_min'], max=physics_denorm_params['xi_0_max'])
        rho_phys = torch.clamp(rho_phys, min=physics_denorm_params['rho_min'], max=physics_denorm_params['rho_max'])
        cohesion_phys = torch.clamp(cohesion_phys, min=physics_denorm_params['cohesion_min'], max=physics_denorm_params['cohesion_max'])
        return {
            : x_for_grad,  
            : x_for_grad[:, 3:4], 
            : x_for_grad[:, 4:5], 
            : x_for_grad[:, 5:6],
            : x_phys, 'y_phys': y_phys, 't_phys': t_phys,
            : h_phys, 'vx_phys': vx_phys, 'vy_phys': vy_phys,
            : dem, 'dzdx': dzdx_phys, 'dzdy': dzdy_phys,  
            : mu_phys, 'xi_phys': xi_phys, 'rho_phys': rho_phys, 'cohesion_phys': cohesion_phys,  
            : coord_scale,
            : coord_scale,
            : time_scale
        }
    def _prepare_gradient_inputs(self, x_batch: torch.Tensor) -> torch.Tensor:
        if x_batch.requires_grad:
            return x_batch
        x_for_grad = x_batch.clone().detach().requires_grad_(True)
        return x_for_grad
    def _compute_active_region_sampling_mask(self, physics_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = physics_data['h_phys']
        if self.enable_active_region_sampling:
            h_flat = h.view(-1)
            active_mask_flat = (h_flat > self.active_height_threshold).float()
            active_points = torch.sum(active_mask_flat)
            if active_points < self.min_active_points:
                logging.info(
                )
                total_points = active_mask_flat.shape[0]
                k = min(int(self.min_active_points), int(total_points))
                k = max(k, 1)  
                topk_vals, topk_idx = torch.topk(h_flat, k)
                sampling_mask_flat = torch.zeros_like(active_mask_flat)
                sampling_mask_flat[topk_idx] = 1.0
                final_mask = sampling_mask_flat.view_as(h)
            else:
                total_points = active_mask_flat.shape[0]
                active_num = int(total_points * self.active_region_sample_ratio)
                inactive_num = int(total_points * self.inactive_region_sample_ratio)
                active_num = min(active_num, int(active_points.item()))
                inactive_candidates = total_points - int(active_points.item())
                inactive_num = min(inactive_num, max(inactive_candidates, 0))
                if active_num > 0:
                    active_indices = torch.nonzero(active_mask_flat > 0, as_tuple=False).squeeze(1)
                    inactive_indices = torch.nonzero(active_mask_flat == 0, as_tuple=False).squeeze(1)
                    if len(active_indices) > active_num:
                        perm = torch.randperm(len(active_indices), device=self.device)
                        sampled_active_indices = active_indices[perm[:active_num]]
                    else:
                        sampled_active_indices = active_indices
                    if len(inactive_indices) > inactive_num:
                        perm = torch.randperm(len(inactive_indices), device=self.device)
                        sampled_inactive_indices = inactive_indices[perm[:inactive_num]]
                    else:
                        sampled_inactive_indices = inactive_indices
                    all_sampled_indices = torch.cat([sampled_active_indices, sampled_inactive_indices])
                    sampling_mask_flat = torch.zeros_like(active_mask_flat)
                    sampling_mask_flat[all_sampled_indices] = 1.0
                    final_mask = sampling_mask_flat.view_as(h)
                    logging.info(
                    )
                else:
                    final_mask = (h > self.active_height_threshold).float()
                    logging.info("Active region sampling not triggered (0 points), using original mask")
        else:
            active_mask = (h > self.active_height_threshold).float()
            active_points = torch.sum(active_mask)
            if active_points < self.min_active_points:
                logging.info(
                )
                h_flat = h.view(-1)
                total_points = h_flat.shape[0]
                k = min(int(self.min_active_points), int(total_points))
                k = max(k, 1)
                _, topk_idx = torch.topk(h_flat, k)
                sampling_mask_flat = torch.zeros_like(h_flat)
                sampling_mask_flat[topk_idx] = 1.0
                final_mask = sampling_mask_flat.view_as(h)
            else:
                final_mask = active_mask
        if torch.sum(final_mask) < self.min_active_points:
            logging.info(f"Insufficient final sampling points: {torch.sum(final_mask)} < {self.min_active_points},returning zero mask")
            return torch.zeros_like(h)
        return final_mask
    def _compute_dimensionless_gradients(self, physics_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        gradients = {}
        try:
            h_phys = physics_data['h_phys']
            vx_phys = physics_data['vx_phys']
            vy_phys = physics_data['vy_phys']
            x_norm = physics_data['x_norm']
            y_norm = physics_data['y_norm']
            t_norm = physics_data['t_norm']
            if not h_phys.requires_grad:
                h_phys.requires_grad_(True)
            if not vx_phys.requires_grad:
                vx_phys.requires_grad_(True)
            if not vy_phys.requires_grad:
                vy_phys.requires_grad_(True)
            coord_scale_x = physics_data['coord_scale_x']
            coord_scale_y = physics_data['coord_scale_y']
            time_scale = physics_data['time_scale']
            source_tensor = physics_data['source_tensor']
            h_grads = torch.autograd.grad(h_phys.sum(), source_tensor, create_graph=True, allow_unused=True, retain_graph=True)[0]
            if h_grads is not None:
                dh_dx_norm = h_grads[:, 3:4]
                dh_dy_norm = h_grads[:, 4:5]
                dh_dt_norm = h_grads[:, 5:6]
            else:
                dh_dx_norm = dh_dy_norm = dh_dt_norm = None
            vx_grads = torch.autograd.grad(vx_phys.sum(), source_tensor, create_graph=True, allow_unused=True, retain_graph=True)[0]
            if vx_grads is not None:
                dvx_dx_norm = vx_grads[:, 3:4]
                dvx_dy_norm = vx_grads[:, 4:5]
                dvx_dt_norm = vx_grads[:, 5:6]
            else:
                dvx_dx_norm = dvx_dy_norm = dvx_dt_norm = None
            vy_grads = torch.autograd.grad(vy_phys.sum(), source_tensor, create_graph=True, allow_unused=True, retain_graph=True)[0]
            if vy_grads is not None:
                dvy_dx_norm = vy_grads[:, 3:4]
                dvy_dy_norm = vy_grads[:, 4:5]
                dvy_dt_norm = vy_grads[:, 5:6]
            else:
                dvy_dx_norm = dvy_dy_norm = dvy_dt_norm = None
            gradients['dh_dt'] = dh_dt_norm / time_scale if dh_dt_norm is not None else torch.zeros_like(h_phys)
            gradients['dh_dx'] = dh_dx_norm / coord_scale_x if dh_dx_norm is not None else torch.zeros_like(h_phys)
            gradients['dh_dy'] = dh_dy_norm / coord_scale_y if dh_dy_norm is not None else torch.zeros_like(h_phys)
            gradients['dvx_dt'] = dvx_dt_norm / time_scale if dvx_dt_norm is not None else torch.zeros_like(vx_phys)
            gradients['dvx_dx'] = dvx_dx_norm / coord_scale_x if dvx_dx_norm is not None else torch.zeros_like(vx_phys)
            gradients['dvx_dy'] = dvx_dy_norm / coord_scale_y if dvx_dy_norm is not None else torch.zeros_like(vx_phys)
            gradients['dvy_dt'] = dvy_dt_norm / time_scale if dvy_dt_norm is not None else torch.zeros_like(vy_phys)
            gradients['dvy_dx'] = dvy_dx_norm / coord_scale_x if dvy_dx_norm is not None else torch.zeros_like(vy_phys)
            gradients['dvy_dy'] = dvy_dy_norm / coord_scale_y if dvy_dy_norm is not None else torch.zeros_like(vy_phys)
            for key, grad in gradients.items():
                gradients[key] = torch.clamp(grad, min=-1e6, max=1e6)
                gradients[key] = torch.where(torch.isnan(grad) | torch.isinf(grad), 
                                           torch.zeros_like(grad), grad)
            return gradients
        except Exception as e:
            logging.error(f"Gradient calculation failed: {e}")
            zero_grad = torch.zeros_like(physics_data['h_phys'])
            return {key: zero_grad for key in ['dh_dt', 'dh_dx', 'dh_dy', 
                                             , 'dvx_dx', 'dvx_dy',
                                             , 'dvy_dx', 'dvy_dy']}
    def _compute_dimensionless_gradients_sampled(self, physics_data: Dict[str, torch.Tensor], 
                                               sampling_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        gradients = {}
        try:
            h_phys = physics_data['h_phys']
            vx_phys = physics_data['vx_phys']
            vy_phys = physics_data['vy_phys']
            x_norm = physics_data['x_norm']
            y_norm = physics_data['y_norm']
            t_norm = physics_data['t_norm']
            if torch.sum(sampling_mask) == 0:
                logging.warning("Sampling mask is empty, returning zero gradient")
                zero_grad = torch.zeros_like(h_phys)
                return {key: zero_grad for key in ['dh_dt', 'dh_dx', 'dh_dy', 
                                                 , 'dvx_dx', 'dvx_dy',
                                                 , 'dvy_dx', 'dvy_dy']}
            if not h_phys.requires_grad:
                h_phys.requires_grad_(True)
            if not vx_phys.requires_grad:
                vx_phys.requires_grad_(True)
            if not vy_phys.requires_grad:
                vy_phys.requires_grad_(True)
            sampled_h = h_phys * sampling_mask
            sampled_vx = vx_phys * sampling_mask
            sampled_vy = vy_phys * sampling_mask
            coord_scale_x = physics_data['coord_scale_x']
            coord_scale_y = physics_data['coord_scale_y']
            time_scale = physics_data['time_scale']
            source_tensor = physics_data['source_tensor']
            h_grads = torch.autograd.grad(sampled_h.sum(), source_tensor, create_graph=True, allow_unused=True, retain_graph=True)[0]
            if h_grads is not None:
                dh_dx_norm = h_grads[:, 3:4]
                dh_dy_norm = h_grads[:, 4:5]
                dh_dt_norm = h_grads[:, 5:6]
            else:
                dh_dx_norm = dh_dy_norm = dh_dt_norm = None
            vx_grads = torch.autograd.grad(sampled_vx.sum(), source_tensor, create_graph=True, allow_unused=True, retain_graph=True)[0]
            if vx_grads is not None:
                dvx_dx_norm = vx_grads[:, 3:4]
                dvx_dy_norm = vx_grads[:, 4:5]
                dvx_dt_norm = vx_grads[:, 5:6]
            else:
                dvx_dx_norm = dvx_dy_norm = dvx_dt_norm = None
            vy_grads = torch.autograd.grad(sampled_vy.sum(), source_tensor, create_graph=True, allow_unused=True, retain_graph=True)[0]
            if vy_grads is not None:
                dvy_dx_norm = vy_grads[:, 3:4]
                dvy_dy_norm = vy_grads[:, 4:5]
                dvy_dt_norm = vy_grads[:, 5:6]
            else:
                dvy_dx_norm = dvy_dy_norm = dvy_dt_norm = None
            gradients['dh_dt'] = (dh_dt_norm / time_scale) * sampling_mask if dh_dt_norm is not None else torch.zeros_like(h_phys)
            gradients['dh_dx'] = (dh_dx_norm / coord_scale_x) * sampling_mask if dh_dx_norm is not None else torch.zeros_like(h_phys)
            gradients['dh_dy'] = (dh_dy_norm / coord_scale_y) * sampling_mask if dh_dy_norm is not None else torch.zeros_like(h_phys)
            gradients['dvx_dt'] = (dvx_dt_norm / time_scale) * sampling_mask if dvx_dt_norm is not None else torch.zeros_like(vx_phys)
            gradients['dvx_dx'] = (dvx_dx_norm / coord_scale_x) * sampling_mask if dvx_dx_norm is not None else torch.zeros_like(vx_phys)
            gradients['dvx_dy'] = (dvx_dy_norm / coord_scale_y) * sampling_mask if dvx_dy_norm is not None else torch.zeros_like(vx_phys)
            gradients['dvy_dt'] = (dvy_dt_norm / time_scale) * sampling_mask if dvy_dt_norm is not None else torch.zeros_like(vy_phys)
            gradients['dvy_dx'] = (dvy_dx_norm / coord_scale_x) * sampling_mask if dvy_dx_norm is not None else torch.zeros_like(vy_phys)
            gradients['dvy_dy'] = (dvy_dy_norm / coord_scale_y) * sampling_mask if dvy_dy_norm is not None else torch.zeros_like(vy_phys)
            for key, grad in gradients.items():
                gradients[key] = torch.clamp(grad, min=-1e6, max=1e6)
                gradients[key] = torch.where(torch.isnan(grad) | torch.isinf(grad), 
                                           torch.zeros_like(grad), grad)
            return gradients
        except Exception as e:
            logging.error(f"Sampling gradient calculation failed: {e}")
            zero_grad = torch.zeros_like(physics_data['h_phys'])
            return {key: zero_grad for key in ['dh_dt', 'dh_dx', 'dh_dy', 
                                             , 'dvx_dx', 'dvx_dy',
                                             , 'dvy_dx', 'dvy_dy']}
    def _compute_dimensionless_pde_residuals(self, physics_data: Dict[str, torch.Tensor], 
                                           gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = physics_data['h_phys']
        vx = physics_data['vx_phys']
        vy = physics_data['vy_phys']
        dzdx = physics_data['dzdx']
        dzdy = physics_data['dzdy']
        flux_x = h * vx  
        flux_y = h * vy  
        dflux_x_dx = gradients['dh_dx'] * vx + h * gradients['dvx_dx']  
        dflux_y_dy = gradients['dh_dy'] * vy + h * gradients['dvy_dy']  
        continuity_residual = gradients['dh_dt'] + dflux_x_dx + dflux_y_dy
        convection_x = vx * gradients['dvx_dx'] + vy * gradients['dvx_dy']  
        convection_y = vx * gradients['dvy_dx'] + vy * gradients['dvy_dy']  
        pressure_x = self.g * gradients['dh_dx']  
        pressure_y = self.g * gradients['dh_dy']  
        terrain_x = self.g * dzdx  
        terrain_y = self.g * dzdy  
        mu_phys = physics_data['mu_phys']  
        xi_phys = physics_data['xi_phys']  
        rho_phys = physics_data['rho_phys']  
        u_mag = torch.sqrt(vx**2 + vy**2 + self.eps)
        u_mag = torch.clamp(u_mag, min=self.eps, max=1e3)  
        n_ux = vx / u_mag
        n_uy = vy / u_mag
        n_ux = torch.where(torch.isnan(n_ux) | torch.isinf(n_ux), torch.zeros_like(n_ux), n_ux)
        n_uy = torch.where(torch.isnan(n_uy) | torch.isinf(n_uy), torch.zeros_like(n_uy), n_uy)
        xi_phys_safe = torch.clamp(xi_phys, min=self.eps, max=1e6)  
        friction_term = mu_phys * self.g * h + self.g * u_mag ** 2 / xi_phys_safe  
        friction_term = torch.clamp(friction_term, min=-1e6, max=1e6)  
        friction_x = n_ux * friction_term  
        friction_y = n_uy * friction_term  
        friction_x = torch.where(torch.isnan(friction_x) | torch.isinf(friction_x), torch.zeros_like(friction_x), friction_x)
        friction_y = torch.where(torch.isnan(friction_y) | torch.isinf(friction_y), torch.zeros_like(friction_y), friction_y)
        momentum_x_residual = gradients['dvx_dt'] + convection_x + pressure_x + terrain_x + friction_x
        momentum_y_residual = gradients['dvy_dt'] + convection_y + pressure_y + terrain_y + friction_y
        if not self.use_dimensionless:
            return {
                : continuity_residual,
                : momentum_x_residual,
                : momentum_y_residual
            }
        T_char = self.dx / self.U_char  
        continuity_scale = self.H_char / T_char  
        continuity_dimensionless = continuity_residual / continuity_scale
        momentum_x_dimensionless = momentum_x_residual / self.A_char
        momentum_y_dimensionless = momentum_y_residual / self.A_char
        return {
            : continuity_dimensionless,
            : momentum_x_dimensionless,
            : momentum_y_dimensionless
        }
    def _compute_dimensionless_pde_residuals_sampled(self, physics_data: Dict[str, torch.Tensor], 
                                                   gradients: Dict[str, torch.Tensor],
                                                   sampling_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = physics_data['h_phys']
        vx = physics_data['vx_phys']
        vy = physics_data['vy_phys']
        dzdx = physics_data['dzdx']
        dzdy = physics_data['dzdy']
        h_sampled = h * sampling_mask
        vx_sampled = vx * sampling_mask
        vy_sampled = vy * sampling_mask
        flux_x = h_sampled * vx_sampled  
        flux_y = h_sampled * vy_sampled  
        dflux_x_dx = gradients['dh_dx'] * vx_sampled + h_sampled * gradients['dvx_dx']  
        dflux_y_dy = gradients['dh_dy'] * vy_sampled + h_sampled * gradients['dvy_dy']  
        continuity_residual = gradients['dh_dt'] + dflux_x_dx + dflux_y_dy
        convection_x = vx_sampled * gradients['dvx_dx'] + vy_sampled * gradients['dvx_dy']  
        convection_y = vx_sampled * gradients['dvy_dx'] + vy_sampled * gradients['dvy_dy']  
        pressure_x = self.g * gradients['dh_dx']  
        pressure_y = self.g * gradients['dh_dy']  
        terrain_x = self.g * dzdx * sampling_mask  
        terrain_y = self.g * dzdy * sampling_mask  
        mu_phys = physics_data['mu_phys'] * sampling_mask  
        xi_phys = physics_data['xi_phys'] * sampling_mask  
        u_mag = torch.sqrt(vx_sampled**2 + vy_sampled**2 + self.eps)
        u_mag = torch.clamp(u_mag, min=self.eps, max=1e3)  
        n_ux = vx_sampled / u_mag
        n_uy = vy_sampled / u_mag
        n_ux = torch.where(torch.isnan(n_ux) | torch.isinf(n_ux), torch.zeros_like(n_ux), n_ux)
        n_uy = torch.where(torch.isnan(n_uy) | torch.isinf(n_uy), torch.zeros_like(n_uy), n_uy)
        xi_phys_safe = torch.clamp(xi_phys, min=self.eps, max=1e6)  
        friction_term = mu_phys * self.g * h_sampled + self.g * u_mag ** 2 / xi_phys_safe  
        friction_term = torch.clamp(friction_term, min=-1e6, max=1e6)  
        friction_x = n_ux * friction_term  
        friction_y = n_uy * friction_term  
        friction_x = torch.where(torch.isnan(friction_x) | torch.isinf(friction_x), torch.zeros_like(friction_x), friction_x)
        friction_y = torch.where(torch.isnan(friction_y) | torch.isinf(friction_y), torch.zeros_like(friction_y), friction_y)
        momentum_x_residual = gradients['dvx_dt'] + convection_x + pressure_x + terrain_x + friction_x
        momentum_y_residual = gradients['dvy_dt'] + convection_y + pressure_y + terrain_y + friction_y
        if not self.use_dimensionless:
            return {
                : continuity_residual,
                : momentum_x_residual,
                : momentum_y_residual
            }
        T_char = self.dx / self.U_char  
        continuity_scale = self.H_char / T_char  
        continuity_dimensionless = continuity_residual / continuity_scale
        momentum_x_dimensionless = momentum_x_residual / self.A_char
        momentum_y_dimensionless = momentum_y_residual / self.A_char
        return {
            : continuity_dimensionless,
            : momentum_x_dimensionless,
            : momentum_y_dimensionless
        }
    def _compute_physics_constraints(self, pde_residuals: Dict[str, torch.Tensor], 
                                   physics_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = physics_data['h_phys']
        if self.enable_active_region_sampling:
            h_flat = h.view(-1)
            active_mask_flat = (h_flat > self.active_height_threshold).float()
            active_points = torch.sum(active_mask_flat)
            if active_points < self.min_active_points:
                logging.info(
                )
                total_points = active_mask_flat.shape[0]
                k = min(int(self.min_active_points), int(total_points))
                k = max(k, 1)  
                topk_vals, topk_idx = torch.topk(h_flat, k)
                sampling_mask_flat = torch.zeros_like(active_mask_flat)
                sampling_mask_flat[topk_idx] = 1.0
                final_mask = sampling_mask_flat.view_as(h)
            else:
                total_points = active_mask_flat.shape[0]
                active_num = int(total_points * self.active_region_sample_ratio)
                inactive_num = int(total_points * self.inactive_region_sample_ratio)
                active_num = min(active_num, int(active_points.item()))
                inactive_candidates = total_points - int(active_points.item())
                inactive_num = min(inactive_num, max(inactive_candidates, 0))
                if active_num > 0:
                    active_indices = torch.nonzero(active_mask_flat > 0, as_tuple=False).squeeze(1)
                    inactive_indices = torch.nonzero(active_mask_flat == 0, as_tuple=False).squeeze(1)
                    if len(active_indices) > active_num:
                        perm = torch.randperm(len(active_indices), device=self.device)
                        sampled_active_indices = active_indices[perm[:active_num]]
                    else:
                        sampled_active_indices = active_indices
                    if len(inactive_indices) > inactive_num:
                        perm = torch.randperm(len(inactive_indices), device=self.device)
                        sampled_inactive_indices = inactive_indices[perm[:inactive_num]]
                    else:
                        sampled_inactive_indices = inactive_indices
                    all_sampled_indices = torch.cat([sampled_active_indices, sampled_inactive_indices])
                    sampling_mask_flat = torch.zeros_like(active_mask_flat)
                    sampling_mask_flat[all_sampled_indices] = 1.0
                    final_mask_flat = active_mask_flat * sampling_mask_flat
                    final_mask = final_mask_flat.view_as(h)
                    logging.info(
                    )
                else:
                    final_mask = (h > self.active_height_threshold).float()
                    logging.info("Active region sampling not triggered (0 points), using original mask")
        else:
            active_mask = (h > self.active_height_threshold).float()
            active_points = torch.sum(active_mask)
            if active_points < self.min_active_points:
                logging.info(
                )
                h_flat = h.view(-1)
                total_points = h_flat.shape[0]
                k = min(int(self.min_active_points), int(total_points))
                k = max(k, 1)
                _, topk_idx = torch.topk(h_flat, k)
                sampling_mask_flat = torch.zeros_like(h_flat)
                sampling_mask_flat[topk_idx] = 1.0
                final_mask = sampling_mask_flat.view_as(h)
            else:
                final_mask = active_mask
        if torch.sum(final_mask) < self.min_active_points:
            logging.info(f"Insufficient final sampling points: {torch.sum(final_mask)} < {self.min_active_points},returning zero loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        continuity_loss = torch.mean(pde_residuals['continuity']**2 * final_mask)
        momentum_x_loss = torch.mean(pde_residuals['momentum_x']**2 * final_mask)
        momentum_y_loss = torch.mean(pde_residuals['momentum_y']**2 * final_mask)
        h_dimensionless = h / self.H_char
        height_constraint = torch.mean((torch.clamp(h_dimensionless, min=0) - h_dimensionless)**2 * final_mask)
        total_physics_loss = (
            self.continuity_weight * continuity_loss +
            self.momentum_weight * (momentum_x_loss + momentum_y_loss) +
            self.height_constraint_weight * height_constraint
        )
        total_physics_loss = torch.clamp(total_physics_loss, min=self.loss_clamp_min, max=self.loss_clamp_max)
        if torch.isnan(total_physics_loss) or torch.isinf(total_physics_loss):
            logging.warning("Physics loss is NaN/Inf, returning zero")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return total_physics_loss
    def _compute_physics_constraints_sampled(self, pde_residuals: Dict[str, torch.Tensor], 
                                           physics_data: Dict[str, torch.Tensor],
                                           sampling_mask: torch.Tensor) -> torch.Tensor:
        h = physics_data['h_phys']
        sampled_points = torch.sum(sampling_mask)
        if sampled_points < self.min_active_points:
            logging.info(f"Insufficient sampled points: {sampled_points} < {self.min_active_points},returning zero loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        continuity_loss = torch.mean(pde_residuals['continuity']**2)
        momentum_x_loss = torch.mean(pde_residuals['momentum_x']**2)
        momentum_y_loss = torch.mean(pde_residuals['momentum_y']**2)
        h_dimensionless = h / self.H_char
        h_sampled = h_dimensionless * sampling_mask
        height_constraint = torch.mean((torch.clamp(h_sampled, min=0) - h_sampled)**2)
        total_physics_loss = (
            self.continuity_weight * continuity_loss +
            self.momentum_weight * (momentum_x_loss + momentum_y_loss) +
            self.height_constraint_weight * height_constraint
        )
        total_physics_loss = torch.clamp(total_physics_loss, min=self.loss_clamp_min, max=self.loss_clamp_max)
        if torch.isnan(total_physics_loss) or torch.isinf(total_physics_loss):
            logging.warning("Physics loss is NaN/Inf, returning zero")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return total_physics_loss
    def compute_physics_loss(self, x_batch, pred_norm, input_norm_stats, target_norm_stats, step=None, 
                           active_sampling_rate=None, boundary_weight=None, model=None):
        try:
            return self.compute_dimensionless_physics_loss(
                x_batch=x_batch,
                predictions=pred_norm if pred_norm is not None else model(x_batch),
                targets=None,  
                pred_norm=pred_norm,
                model=model  
            )
        except Exception as e:
            logging.warning(f"Dimensionless physics loss calculation failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    def compute_boundary_loss(self, model, x_batch, input_norm_stats, target_norm_stats):
        if not self.enable_boundary_condition_loss:
            return torch.tensor(0.0, device=self.device)
        try:
            if model is None:
                return torch.tensor(0.0, device=x_batch.device, requires_grad=True)
            h_norm = x_batch[:, 0:1]      
            vx_norm = x_batch[:, 1:2]     
            vy_norm = x_batch[:, 2:3]     
            x_norm = x_batch[:, 3:4]      
            y_norm = x_batch[:, 4:5]      
            t_norm = x_batch[:, 5:6]      
            release_mask = x_batch[:, 9:10]  
            predictions = model(x_batch)
            height_mean, height_std = self.global_data_config.get_height_denorm_params()
            pred_h_orig = predictions[:, 0:1] * height_std + height_mean
            velocity_params = self.global_data_config.get_velocity_denorm_params()
            vx_mean = velocity_params['velocity_x_mean']
            vx_std = velocity_params['velocity_x_std']
            vy_mean = velocity_params['velocity_y_mean']
            vy_std = velocity_params['velocity_y_std']
            pred_vx_orig = predictions[:, 1:2] * vx_std + vx_mean
            pred_vy_orig = predictions[:, 2:3] * vy_std + vy_mean
            boundary_loss = torch.tensor(0.0, device=self.device)
            t_zero_mask = (t_norm < 0.01)  
            if t_zero_mask.any():
                non_release_mask = (release_mask < 0.5) & t_zero_mask
                if non_release_mask.any():
                    initial_height_loss = torch.mean((pred_h_orig[non_release_mask])**2)
                    boundary_loss += 0.1 * initial_height_loss  
                if t_zero_mask.any():
                    velocity_threshold = 0.5  
                    excess_vx = torch.relu(torch.abs(pred_vx_orig[t_zero_mask]) - velocity_threshold)
                    excess_vy = torch.relu(torch.abs(pred_vy_orig[t_zero_mask]) - velocity_threshold)
                    initial_velocity_loss = torch.mean(excess_vx**2) + torch.mean(excess_vy**2)
                    boundary_loss += 0.5 * initial_velocity_loss  
            boundary_tolerance = 0.1
            left_boundary = (x_norm < boundary_tolerance)
            if left_boundary.any():
                excessive_vx = torch.relu(pred_vx_orig[left_boundary] - 2.0)
                left_boundary_loss = torch.mean(excessive_vx**2)
                boundary_loss += 0.1 * left_boundary_loss  
            right_boundary = (x_norm > (1.0 - boundary_tolerance))
            if right_boundary.any():
                excessive_vx = torch.relu(-pred_vx_orig[right_boundary] - 2.0)
                right_boundary_loss = torch.mean(excessive_vx**2)
                boundary_loss += 0.1 * right_boundary_loss  
            top_boundary = (y_norm > (1.0 - boundary_tolerance))
            if top_boundary.any():
                excessive_vy = torch.relu(-pred_vy_orig[top_boundary] - 2.0)
                top_boundary_loss = torch.mean(excessive_vy**2)
                boundary_loss += 0.1 * top_boundary_loss  
            bottom_boundary = (y_norm < boundary_tolerance)
            if bottom_boundary.any():
                excessive_vy = torch.relu(pred_vy_orig[bottom_boundary] - 2.0)
                bottom_boundary_loss = torch.mean(excessive_vy**2)
                boundary_loss += 0.1 * bottom_boundary_loss  
            release_boundary = (release_mask > 0.5) & (t_norm < 0.02)  
            if release_boundary.any():
                min_release_height = 0.5  
                max_release_height = 8.0  
                too_low = torch.relu(min_release_height - pred_h_orig[release_boundary])
                too_high = torch.relu(pred_h_orig[release_boundary] - max_release_height)
                release_height_loss = torch.mean(too_low**2) + torch.mean(too_high**2)
                boundary_loss += 0.3 * release_height_loss  
            if torch.isnan(boundary_loss):
                logging.warning(f"Boundary loss contains NaN, returning zero")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            if torch.isinf(boundary_loss):
                logging.warning(f"Boundary loss contains Inf, returning zero")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            boundary_loss = torch.clamp(boundary_loss, min=self.loss_clamp_min, max=self.loss_clamp_max)
            return boundary_loss
        except Exception as e:
            logging.warning(f"Boundary condition loss calculation failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    def compute_initial_condition_loss(self, model, x_batch, input_norm_stats, target_norm_stats):
        if not self.enable_initial_condition_loss:
            return torch.tensor(0.0, device=self.device)
        try:
            h_norm = x_batch[:, 0:1]      
            vx_norm = x_batch[:, 1:2]     
            vy_norm = x_batch[:, 2:3]     
            t_norm = x_batch[:, 5:6]      
            release_mask = x_batch[:, 9:10]  
            predictions = model(x_batch)
            height_mean, height_std = self.global_data_config.get_height_denorm_params()
            pred_h_orig = predictions[:, 0:1] * height_std + height_mean
            velocity_params = self.global_data_config.get_velocity_denorm_params()
            vx_mean = velocity_params['velocity_x_mean']
            vx_std = velocity_params['velocity_x_std']
            vy_mean = velocity_params['velocity_y_mean']
            vy_std = velocity_params['velocity_y_std']
            pred_vx_orig = predictions[:, 1:2] * vx_std + vx_mean
            pred_vy_orig = predictions[:, 2:3] * vy_std + vy_mean
            initial_condition_tolerance = 0.02  
            is_initial_time = (t_norm < initial_condition_tolerance)
            initial_loss = torch.tensor(0.0, device=self.device)
            if torch.sum(is_initial_time) > 0:
                initial_vx = pred_vx_orig[is_initial_time]
                initial_vy = pred_vy_orig[is_initial_time]
                velocity_magnitude = torch.sqrt(initial_vx**2 + initial_vy**2 + self.eps)
                velocity_loss = torch.mean(torch.relu(velocity_magnitude - 0.5)**2)  
                initial_release_mask = release_mask[is_initial_time]
                initial_height = pred_h_orig[is_initial_time]
                is_release_area = (initial_release_mask > 0.5)
                if torch.sum(is_release_area) > 0:
                    release_height = initial_height[is_release_area]
                    min_release_height = 0.5
                    max_release_height = 8.0
                    too_low = torch.relu(min_release_height - release_height)
                    too_high = torch.relu(release_height - max_release_height)
                    release_height_loss = torch.mean(too_low**2 + too_high**2)
                else:
                    release_height_loss = torch.tensor(0.0, device=self.device)
                is_non_release_area = (initial_release_mask <= 0.5)
                if torch.sum(is_non_release_area) > 0:
                    non_release_height = initial_height[is_non_release_area]
                    non_release_height_loss = torch.mean(torch.relu(non_release_height - 0.1)**2)  
                else:
                    non_release_height_loss = torch.tensor(0.0, device=self.device)
                height_regularization = torch.tensor(0.0, device=self.device)
                if initial_height.numel() > 1:
                    height_flat = initial_height.view(-1)
                    if len(height_flat) > 1:
                        height_diff = torch.diff(height_flat)
                        height_regularization = torch.mean(height_diff**2) * 0.1  
                negative_height_penalty = torch.mean(torch.relu(-initial_height)**2) * 2.0
                max_physical_height = 50.0  
                excessive_height_penalty = torch.mean(torch.relu(initial_height - max_physical_height)**2) * 1.5
                initial_loss = (
                    1.0 * velocity_loss +
                    0.5 * release_height_loss +
                    0.3 * non_release_height_loss +
                    0.2 * height_regularization +      
                    0.4 * negative_height_penalty +    
                    0.3 * excessive_height_penalty     
                )
            if torch.isnan(initial_loss):
                logging.warning(f"Initial condition loss contains NaN, returning zero")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            if torch.isinf(initial_loss):
                logging.warning(f"Initial condition loss contains Inf, returning zero")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            initial_loss = torch.clamp(initial_loss, min=self.loss_clamp_min, max=self.loss_clamp_max)
            return initial_loss
        except Exception as e:
            logging.warning(f"Initial condition loss calculation failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    def update_weights(self, epoch, max_epochs):
        progress = epoch / max_epochs
        if progress < 0.3:
            self.eq1_weight = 1.0
            self.eq2_weight = 0.8
            self.eq3_weight = 0.8
        elif progress < 0.7:
            self.eq1_weight = 1.5
            self.eq2_weight = 1.0
            self.eq3_weight = 1.0
        else:
            self.eq1_weight = 2.0
            self.eq2_weight = 1.2
            self.eq3_weight = 1.2
def create_dimensionless_physics_loss(config: Dict[str, Any], global_data_config, device='cuda') -> DimensionlessPhysicsLoss:
    return DimensionlessPhysicsLoss(config, global_data_config, device)