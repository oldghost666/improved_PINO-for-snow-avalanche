
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
        self._initialize_characteristic_scales()
        self.eps = 1e-8
        self.loss_clamp_min = 1e-8
        self.loss_clamp_max = 1e6
        self.continuity_weight = 1.0
        self.momentum_weight = 1.0
        self.height_constraint_weight = 1.0
        pl_cfg = self.config.get('physics_loss', {})
        self.continuity_weight = float(pl_cfg.get('continuity_weight', self.continuity_weight))
        self.momentum_x_weight = float(pl_cfg.get('momentum_x_weight', self.momentum_weight))
        self.momentum_y_weight = float(pl_cfg.get('momentum_y_weight', self.momentum_weight))
        self.height_constraint_weight = float(pl_cfg.get('height_constraint_weight', self.height_constraint_weight))
        self.boundary_constraint_weight = float(pl_cfg.get('boundary_constraint_weight', 0.0))
        self.initial_constraint_weight = float(pl_cfg.get('initial_constraint_weight', 0.0))
        ic_cfg = self.config.get('physics', {}).get('initial_conditions', {})
        self.initial_velocity_threshold = float(ic_cfg.get('velocity_threshold', 0.5))
        hr = ic_cfg.get('height_range', [0.5, 8.0])
        self.release_height_min = float(hr[0] if isinstance(hr, (list, tuple)) and len(hr) > 0 else 0.5)
        self.release_height_max = float(hr[1] if isinstance(hr, (list, tuple)) and len(hr) > 1 else 8.0)
        self.non_release_height_threshold = float(ic_cfg.get('non_release_height_threshold', 0.1))
        self.height_max_limit = float(ic_cfg.get('height_max_limit', 50.0))
        bc_cfg = self.config.get('physics', {}).get('boundary_conditions', {})
        self.inflow_velocity_threshold = float(bc_cfg.get('inflow_velocity_threshold', 2.0))
        self.eq1_weight = self.continuity_weight  
        self.eq2_weight = self.momentum_weight    
        self.eq3_weight = self.momentum_weight    
        physics_cfg = self.config.get('physics', {})
        self.active_height_threshold = float(physics_cfg.get('active_height_threshold', 0.05))
        self.default_active_sampling_rate = float(physics_cfg.get('active_region_sample_ratio', 1.0))
        self.zero_grad_count = 0
        logging.info(f"Dimensionless physics loss initialized:")
        logging.info(f"  Spatial scale: dx={self.dx}m, dy={self.dy}m")
        logging.info(f"  Time scale: dt={self.dt}s")
        logging.info(f"  Characteristic height scale: H_char={self.H_char:.3f}m")
        logging.info(f"  Characteristic velocity scale: U_char={self.U_char:.3f}m/s")
        logging.info(f"  Characteristic acceleration scale: A_char={self.A_char:.3f}m/s2")
    def _prepare_model_input(self, x_batch: torch.Tensor, 
                           static_data: Optional[torch.Tensor] = None,
                           physics_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        if static_data is None and physics_params is None:
            return x_batch
        inputs = [x_batch]
        if static_data is not None:
            if len(static_data.shape) != len(x_batch.shape):
                static_data = static_data.expand_as(x_batch[:, :static_data.shape[1]])
            inputs.append(static_data)
        if physics_params is not None:
            if len(physics_params.shape) != len(x_batch.shape):
                physics_params = physics_params.expand_as(x_batch[:, :physics_params.shape[1]])
            inputs.append(physics_params)
        return torch.cat(inputs, dim=1)
    def forward(self, predictions: torch.Tensor, x_batch: torch.Tensor, 
                targets: Optional[torch.Tensor] = None, 
                pred_norm: Optional[torch.Tensor] = None,
                model: Optional[torch.nn.Module] = None,
                static_data: Optional[torch.Tensor] = None,
                physics_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.compute_dimensionless_physics_loss(
            x_batch=x_batch,
            predictions=pred_norm if pred_norm is not None else (model(x_batch) if model is not None else predictions),
            targets=None,  
            model=model,  
            active_sampling_rate=self.default_active_sampling_rate,  
            static_data=static_data,
            physics_params=physics_params
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
            try:
                stats = getattr(self.global_data_config, 'stats', {})
                h_q95 = float(stats.get('height_q95', 0.0))
                u_q95 = float(stats.get('velocity_q95', 0.0))
            except Exception:
                h_q95 = 0.0
                u_q95 = 0.0
            self.H_char = max(h_q95, abs(self.H_mean) + 2.0 * self.H_std, 0.1)
            self.U_char = max(u_q95, self.U_mean + 2.0 * self.U_std, np.sqrt(self.g * self.H_char), 1.0)
            self.A_char = self.g  
            self.Froude = self.U_char / max(1e-8, np.sqrt(self.g * self.H_char))  
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
                                         targets: Optional[torch.Tensor] = None,
                                         model: Optional[torch.nn.Module] = None,
                                         active_sampling_rate: Optional[float] = None,
                                         static_data: Optional[torch.Tensor] = None,
                                         physics_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            if model is not None:
                model.train()
                x_for_physics = self._prepare_gradient_inputs(x_batch)
                model_input = self._prepare_model_input(x_for_physics, static_data, physics_params)
                if not model_input.requires_grad:
                    model_input = model_input.requires_grad_(True)
                predictions_with_grad = model(model_input)
                if not predictions_with_grad.requires_grad:
                    logging.error(f"Model prediction tensor missing gradients: requires_grad={predictions_with_grad.requires_grad}, grad_fn={predictions_with_grad.grad_fn}")
                    predictions_with_grad = predictions_with_grad.requires_grad_(True)
                physics_data = self._prepare_physics_data(x_for_physics, predictions_with_grad)
            else:
                if not predictions.requires_grad:
                    predictions = predictions.detach().requires_grad_(True)
                x_for_grad = self._prepare_gradient_inputs(x_batch)
                physics_data = self._prepare_physics_data(x_for_grad, predictions)
            gradients = self._compute_dimensionless_gradients(physics_data)
            pde_residuals = self._compute_dimensionless_pde_residuals(physics_data, gradients)
            physics_loss = self._compute_physics_constraints(
                pde_residuals, physics_data,
                active_sampling_rate=active_sampling_rate
            )
            return physics_loss
        except Exception as e:
            logging.error(f"Dimensionless physics loss calculation failed: {e}")
            logging.error(f"Error details: {str(e)}")
            import traceback
            logging.error(f"Stack trace: {traceback.format_exc()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    def _prepare_physics_data(self, x_batch: torch.Tensor, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        if len(x_batch.shape) == 4:  
            batch_size, channels, height, width = x_batch.shape
            x_batch_flat = x_batch.permute(0, 2, 3, 1).reshape(-1, channels)
            predictions_flat = predictions.permute(0, 2, 3, 1).reshape(-1, 3)
        else:  
            x_batch_flat = x_batch
            predictions_flat = predictions
        if not predictions_flat.requires_grad:
            logging.error(f"Height prediction tensor missing gradient attribute or computation graph,requires_grad: {predictions_flat.requires_grad}, grad_fn: {predictions_flat.grad_fn}")
            if predictions_flat.grad_fn is None:
                logging.error("Gradient calculation failed: Height prediction tensor missing correct computation graph connection")
                raise RuntimeError("Prediction tensor missing computation graph connection, cannot compute gradient")
        batch_size = x_batch_flat.shape[0]
        x_for_grad = x_batch_flat[:, 3:4]  
        y_for_grad = x_batch_flat[:, 4:5]  
        t_for_grad = x_batch_flat[:, 5:6]  
        if not x_for_grad.requires_grad:
            logging.debug(f"Warning: x-coordinate tensor missing gradient attribute, original tensor gradient status:{x_batch_flat.requires_grad}")
            x_for_grad = x_for_grad.requires_grad_(True)
        if not y_for_grad.requires_grad:
            logging.debug(f"Warning: y-coordinate tensor missing gradient attribute, original tensor gradient status:{x_batch_flat.requires_grad}")
            y_for_grad = y_for_grad.requires_grad_(True)
        if not t_for_grad.requires_grad:
            logging.debug(f"Warning: t-coordinate tensor missing gradient attribute, original tensor gradient status:{x_batch_flat.requires_grad}")
            t_for_grad = t_for_grad.requires_grad_(True)
        coord_scale = self.global_data_config.get_coord_scale()
        time_scale = self.global_data_config.get_time_scale()
        coord_scale_tensor = torch.tensor(coord_scale, device=x_for_grad.device, dtype=x_for_grad.dtype)
        time_scale_tensor = torch.tensor(time_scale, device=t_for_grad.device, dtype=t_for_grad.dtype)
        x_phys = x_for_grad * coord_scale_tensor  
        y_phys = y_for_grad * coord_scale_tensor  
        t_phys = t_for_grad * time_scale_tensor  
        h_mean, h_std = self.global_data_config.get_height_denorm_params()
        velocity_params = self.global_data_config.get_velocity_denorm_params()
        h_mean_tensor = torch.tensor(h_mean, device=predictions_flat.device, dtype=predictions_flat.dtype)
        h_std_tensor = torch.tensor(h_std, device=predictions_flat.device, dtype=predictions_flat.dtype)
        vx_mean_tensor = torch.tensor(velocity_params['velocity_x_mean'], device=predictions_flat.device, dtype=predictions_flat.dtype)
        vx_std_tensor = torch.tensor(velocity_params['velocity_x_std'], device=predictions_flat.device, dtype=predictions_flat.dtype)
        vy_mean_tensor = torch.tensor(velocity_params['velocity_y_mean'], device=predictions_flat.device, dtype=predictions_flat.dtype)
        vy_std_tensor = torch.tensor(velocity_params['velocity_y_std'], device=predictions_flat.device, dtype=predictions_flat.dtype)
        h_phys = predictions_flat[:, 0:1] * h_std_tensor + h_mean_tensor  
        vx_phys = predictions_flat[:, 1:2] * vx_std_tensor + vx_mean_tensor  
        vy_phys = predictions_flat[:, 2:3] * vy_std_tensor + vy_mean_tensor  
        dem = x_batch_flat[:, 6:7]  
        dzdx_norm = x_batch_flat[:, 7:8]  
        dzdy_norm = x_batch_flat[:, 8:9]  
        release_mask = x_batch_flat[:, 9:10]
        gradient_denorm_params = self.global_data_config.get_gradient_denorm_params()
        dzdx_phys = dzdx_norm * gradient_denorm_params['dzdx_std'] + gradient_denorm_params['dzdx_mean']
        dzdy_phys = dzdy_norm * gradient_denorm_params['dzdy_std'] + gradient_denorm_params['dzdy_mean']
        if x_batch_flat.shape[1] != 14:
            raise ValueError(f"Expected 14-channel input, but got{x_batch_flat.shape[1]}channels. Ensure input contains all physical parameters.")
        mu_norm = x_batch_flat[:, 10:11]      
        xi_norm = x_batch_flat[:, 11:12]      
        rho_norm = x_batch_flat[:, 12:13]     
        cohesion_norm = x_batch_flat[:, 13:14] 
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
            : x_for_grad, 'y_norm': y_for_grad, 't_norm': t_for_grad,
            : x_phys, 'y_phys': y_phys, 't_phys': t_phys,
            : h_phys, 'vx_phys': vx_phys, 'vy_phys': vy_phys,
            : dem, 'dzdx': dzdx_phys, 'dzdy': dzdy_phys,  
            : mu_phys, 'xi_phys': xi_phys, 'rho_phys': rho_phys, 'cohesion_phys': cohesion_phys,  
            : coord_scale,
            : coord_scale,
            : time_scale,
            : release_mask
        }
    def _prepare_gradient_inputs(self, x_batch: torch.Tensor) -> torch.Tensor:
        if x_batch.requires_grad:
            return x_batch
        else:
            return x_batch.requires_grad_(True)
    def _compute_dimensionless_gradients(self, physics_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        gradients = {}
        try:
            h_phys = physics_data['h_phys']
            vx_phys = physics_data['vx_phys']
            vy_phys = physics_data['vy_phys']
            x_norm = physics_data['x_norm']
            y_norm = physics_data['y_norm']
            t_norm = physics_data['t_norm']
            if not h_phys.requires_grad or h_phys.grad_fn is None:
                logging.error(f"Height prediction tensor missing gradient attribute or computation graph,requires_grad: {h_phys.requires_grad}, grad_fn: {h_phys.grad_fn}")
                raise RuntimeError("Height prediction tensor missing correct computation graph connection")
            if not vx_phys.requires_grad or vx_phys.grad_fn is None:
                logging.error(f"x-velocity prediction tensor missing gradient attribute or computation graph,requires_grad: {vx_phys.requires_grad}, grad_fn: {vx_phys.grad_fn}")
                raise RuntimeError("x-velocity prediction tensor missing correct computation graph connection")
            if not vy_phys.requires_grad or vy_phys.grad_fn is None:
                logging.error(f"y-velocity prediction tensor missing gradient attribute or computation graph,requires_grad: {vy_phys.requires_grad}, grad_fn: {vy_phys.grad_fn}")
                raise RuntimeError("y-velocity prediction tensor missing correct computation graph connection")
            dh_dt_norm = torch.autograd.grad(h_phys.sum(), t_norm, create_graph=True, allow_unused=True, retain_graph=True)[0]
            dh_dx_norm = torch.autograd.grad(h_phys.sum(), x_norm, create_graph=True, allow_unused=True, retain_graph=True)[0]
            dh_dy_norm = torch.autograd.grad(h_phys.sum(), y_norm, create_graph=True, allow_unused=True, retain_graph=True)[0]
            dvx_dt_norm = torch.autograd.grad(vx_phys.sum(), t_norm, create_graph=True, allow_unused=True, retain_graph=True)[0]
            dvx_dx_norm = torch.autograd.grad(vx_phys.sum(), x_norm, create_graph=True, allow_unused=True, retain_graph=True)[0]
            dvx_dy_norm = torch.autograd.grad(vx_phys.sum(), y_norm, create_graph=True, allow_unused=True, retain_graph=True)[0]
            dvy_dt_norm = torch.autograd.grad(vy_phys.sum(), t_norm, create_graph=True, allow_unused=True, retain_graph=True)[0]
            dvy_dx_norm = torch.autograd.grad(vy_phys.sum(), x_norm, create_graph=True, allow_unused=True, retain_graph=True)[0]
            dvy_dy_norm = torch.autograd.grad(vy_phys.sum(), y_norm, create_graph=True, allow_unused=True, retain_graph=True)[0]
            gradients['dh_dt'] = dh_dt_norm / self.dt if dh_dt_norm is not None else torch.zeros_like(h_phys)
            gradients['dh_dx'] = dh_dx_norm / self.dx if dh_dx_norm is not None else torch.zeros_like(h_phys)
            gradients['dh_dy'] = dh_dy_norm / self.dy if dh_dy_norm is not None else torch.zeros_like(h_phys)
            gradients['dvx_dt'] = dvx_dt_norm / self.dt if dvx_dt_norm is not None else torch.zeros_like(vx_phys)
            gradients['dvx_dx'] = dvx_dx_norm / self.dx if dvx_dx_norm is not None else torch.zeros_like(vx_phys)
            gradients['dvx_dy'] = dvx_dy_norm / self.dy if dvx_dy_norm is not None else torch.zeros_like(vx_phys)
            gradients['dvy_dt'] = dvy_dt_norm / self.dt if dvy_dt_norm is not None else torch.zeros_like(vy_phys)
            gradients['dvy_dx'] = dvy_dx_norm / self.dx if dvy_dx_norm is not None else torch.zeros_like(vy_phys)
            gradients['dvy_dy'] = dvy_dy_norm / self.dy if dvy_dy_norm is not None else torch.zeros_like(vy_phys)
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
        T_char = self.dx / self.U_char  
        continuity_scale = self.H_char / T_char  
        continuity_dimensionless = continuity_residual / continuity_scale
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
        momentum_x_dimensionless = momentum_x_residual / self.A_char
        momentum_y_dimensionless = momentum_y_residual / self.A_char
        try:
            cont_mean = float(torch.mean(continuity_dimensionless).item())
            cont_std = float(torch.std(continuity_dimensionless).item())
            mx_mean = float(torch.mean(momentum_x_dimensionless).item())
            mx_std = float(torch.std(momentum_x_dimensionless).item())
            my_mean = float(torch.mean(momentum_y_dimensionless).item())
            my_std = float(torch.std(momentum_y_dimensionless).item())
            logging.info(
            )
        except Exception:
            pass
        return {
            : continuity_dimensionless,
            : momentum_x_dimensionless,
            : momentum_y_dimensionless
        }
    def compute_residual_loss_on_points(
        self,
        states: Dict[str, torch.Tensor],
        grads: Dict[str, torch.Tensor],
        static: Dict[str, torch.Tensor],
        active_mask: Optional[torch.Tensor] = None,
        sample_ratio: Optional[float] = None
    ) -> torch.Tensor:
        try:
            h = states['h']; vx = states['vx']; vy = states['vy']
            dh_dt = grads['dh_dt']; dh_dx = grads['dh_dx']; dh_dy = grads['dh_dy']
            dvx_dt = grads['dvx_dt']; dvx_dx = grads['dvx_dx']; dvx_dy = grads['dvx_dy']
            dvy_dt = grads['dvy_dt']; dvy_dx = grads['dvy_dx']; dvy_dy = grads['dvy_dy']
            dzdx = static.get('dzdx', torch.zeros_like(h)); dzdy = static.get('dzdy', torch.zeros_like(h))
            mu_phys = static.get('mu', torch.zeros_like(h))
            xi_phys = static.get('xi', torch.ones_like(h))
            rho_phys = static.get('rho', torch.full_like(h, 200.0))
            boundary_mask = static.get('boundary_mask', None)
            initial_mask = static.get('initial_mask', None)
            release_mask = static.get('release_mask', None)
            dflux_x_dx = dh_dx * vx + h * dvx_dx
            dflux_y_dy = dh_dy * vy + h * dvy_dy
            continuity_residual = dh_dt + dflux_x_dx + dflux_y_dy
            T_char = self.dx / self.U_char
            continuity_scale = self.H_char / T_char
            continuity_dimensionless = continuity_residual / continuity_scale
            convection_x = vx * dvx_dx + vy * dvx_dy
            convection_y = vx * dvy_dx + vy * dvy_dy
            pressure_x = self.g * dh_dx
            pressure_y = self.g * dh_dy
            terrain_x = self.g * dzdx
            terrain_y = self.g * dzdy
            u_mag = torch.sqrt(vx**2 + vy**2 + self.eps)
            u_mag = torch.clamp(u_mag, min=self.eps, max=1e3)
            n_ux = torch.where(torch.isnan(vx/u_mag) | torch.isinf(vx/u_mag), torch.zeros_like(vx), vx/u_mag)
            n_uy = torch.where(torch.isnan(vy/u_mag) | torch.isinf(vy/u_mag), torch.zeros_like(vy), vy/u_mag)
            xi_phys_safe = torch.clamp(xi_phys, min=self.eps, max=1e6)
            friction_term = mu_phys * self.g * h + self.g * (u_mag ** 2) / xi_phys_safe
            friction_term = torch.clamp(friction_term, min=-1e6, max=1e6)
            friction_x = torch.where(torch.isnan(n_ux * friction_term) | torch.isinf(n_ux * friction_term), torch.zeros_like(h), n_ux * friction_term)
            friction_y = torch.where(torch.isnan(n_uy * friction_term) | torch.isinf(n_uy * friction_term), torch.zeros_like(h), n_uy * friction_term)
            momentum_x_residual = dvx_dt + convection_x + pressure_x + terrain_x + friction_x
            momentum_y_residual = dvy_dt + convection_y + pressure_y + terrain_y + friction_y
            momentum_x_dimensionless = momentum_x_residual / self.A_char
            momentum_y_dimensionless = momentum_y_residual / self.A_char
            if active_mask is None:
                active_mask = (h > float(getattr(self, 'active_height_threshold', 0.05))).float()
            active_mask = active_mask.view_as(h)
            if int(torch.sum(active_mask).item()) < 1:
                return torch.tensor(0.0, device=h.device, requires_grad=True)
            cont_sq = (continuity_dimensionless ** 2)
            mx_sq = (momentum_x_dimensionless ** 2)
            my_sq = (momentum_y_dimensionless ** 2)
            if sample_ratio is not None:
                sr = float(max(0.0, min(1.0, sample_ratio)))
                active_flat = active_mask.view(-1)
                idxs = torch.nonzero(active_flat, as_tuple=False).squeeze(-1)
                count = max(1, int(idxs.shape[0] * sr))
                perm = torch.randperm(idxs.shape[0], device=h.device)
                sel = idxs[perm[:count]]
                cont_sq_flat = cont_sq.view(-1)
                mx_sq_flat = mx_sq.view(-1)
                my_sq_flat = my_sq.view(-1)
                continuity_loss = cont_sq_flat.index_select(0, sel).mean()
                momentum_x_loss = mx_sq_flat.index_select(0, sel).mean()
                momentum_y_loss = my_sq_flat.index_select(0, sel).mean()
            else:
                continuity_loss = torch.mean(cont_sq * active_mask)
                momentum_x_loss = torch.mean(mx_sq * active_mask)
                momentum_y_loss = torch.mean(my_sq * active_mask)
            h_dimensionless = h / self.H_char
            height_constraint = torch.mean((torch.clamp(h_dimensionless, min=0) - h_dimensionless)**2)
            boundary_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)
            initial_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)
            if self.enable_boundary_condition_loss and boundary_mask is not None:
                vel_mag = torch.sqrt(vx**2 + vy**2 + self.eps)
                excessive_v = torch.relu(vel_mag - self.inflow_velocity_threshold)
                boundary_loss = torch.mean((excessive_v**2) * boundary_mask)
            if self.enable_initial_condition_loss and initial_mask is not None:
                vel_mag_init = torch.sqrt(vx**2 + vy**2 + self.eps)
                init_velocity_excess = torch.relu(vel_mag_init - self.initial_velocity_threshold)
                init_velocity_loss = torch.mean((init_velocity_excess**2) * initial_mask)
                if release_mask is not None:
                    is_release_pts = (release_mask > 0.5).float() * initial_mask
                    is_non_release_pts = (release_mask <= 0.5).float() * initial_mask
                else:
                    is_release_pts = torch.zeros_like(initial_mask)
                    is_non_release_pts = initial_mask
                release_h_too_low = torch.relu(self.release_height_min - h)
                release_h_too_high = torch.relu(h - self.release_height_max)
                release_height_loss = torch.mean(((release_h_too_low**2) + (release_h_too_high**2)) * is_release_pts)
                non_release_height_excess = torch.relu(h - self.non_release_height_threshold)
                non_release_height_loss = torch.mean((non_release_height_excess**2) * is_non_release_pts)
                negative_height_penalty = torch.mean(torch.relu(-h)**2 * initial_mask)
                excessive_height_penalty = torch.mean(torch.relu(h - self.height_max_limit)**2 * initial_mask)
                initial_loss = (
                    1.0 * init_velocity_loss +
                    0.5 * release_height_loss +
                    0.3 * non_release_height_loss +
                    0.4 * negative_height_penalty +
                    0.3 * excessive_height_penalty
                )
            total_physics_loss = (
                self.continuity_weight * continuity_loss +
                self.momentum_x_weight * momentum_x_loss +
                self.momentum_y_weight * momentum_y_loss +
                self.height_constraint_weight * height_constraint +
                self.boundary_constraint_weight * boundary_loss +
                self.initial_constraint_weight * initial_loss
            )
            total_physics_loss = torch.clamp(total_physics_loss, min=self.loss_clamp_min, max=self.loss_clamp_max)
            return total_physics_loss
        except Exception as e:
            logging.warning(f"Point-level Physics loss calculation failed: {e}")
            return torch.tensor(0.0, device=states['h'].device, requires_grad=True)
    def _compute_physics_constraints(self, pde_residuals: Dict[str, torch.Tensor], 
                                   physics_data: Dict[str, torch.Tensor],
                                   active_sampling_rate: Optional[float] = None) -> torch.Tensor:
        h = physics_data['h_phys']
        active_mask = (h > self.active_height_threshold).float()
        active_points = int(torch.sum(active_mask).item())
        if active_points < 10:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        sample_ratio = self.default_active_sampling_rate if active_sampling_rate is None else float(active_sampling_rate)
        sample_ratio = float(max(0.0, min(1.0, sample_ratio)))
        sample_count = max(1, int(active_points * sample_ratio))
        active_mask_flat = active_mask.reshape(-1)
        active_indices = torch.nonzero(active_mask_flat, as_tuple=False).squeeze(-1)
        perm = torch.randperm(active_indices.shape[0], device=self.device)
        selected = active_indices[perm[:sample_count]]
        continuity_sq = (pde_residuals['continuity'] ** 2).reshape(-1)
        momentum_x_sq = (pde_residuals['momentum_x'] ** 2).reshape(-1)
        momentum_y_sq = (pde_residuals['momentum_y'] ** 2).reshape(-1)
        continuity_loss = continuity_sq.index_select(0, selected).mean()
        momentum_x_loss = momentum_x_sq.index_select(0, selected).mean()
        momentum_y_loss = momentum_y_sq.index_select(0, selected).mean()
        h_dimensionless = h / self.H_char
        height_constraint = torch.mean((torch.clamp(h_dimensionless, min=0) - h_dimensionless)**2)
        boundary_loss = torch.tensor(0.0, device=self.device, dtype=h.dtype)
        initial_loss = torch.tensor(0.0, device=self.device, dtype=h.dtype)
        if self.enable_boundary_condition_loss:
            x_norm = physics_data['x_norm']; y_norm = physics_data['y_norm']
            vx = physics_data['vx_phys']; vy = physics_data['vy_phys']
            tol = 0.05
            left = (x_norm < tol)
            right = (x_norm > (1.0 - tol))
            bottom = (y_norm < tol)
            top = (y_norm > (1.0 - tol))
            thr = self.inflow_velocity_threshold
            left_excess = torch.relu(vx - thr)
            right_excess = torch.relu(-vx - thr)
            bottom_excess = torch.relu(vy - thr)
            top_excess = torch.relu(-vy - thr)
            h = physics_data['h_phys']; rx = physics_data.get('release_mask', torch.zeros_like(h))
            non_release_boundary = ((left | right | bottom | top) & (rx < 0.5))
            boundary_height_loss = (h[non_release_boundary].abs().pow(2).mean() if non_release_boundary.any() else torch.tensor(0.0, device=self.device))
            boundary_loss = (
                (left_excess[left].pow(2).mean() if left.any() else torch.tensor(0.0, device=self.device)) +
                (right_excess[right].pow(2).mean() if right.any() else torch.tensor(0.0, device=self.device)) +
                (bottom_excess[bottom].pow(2).mean() if bottom.any() else torch.tensor(0.0, device=self.device)) +
                (top_excess[top].pow(2).mean() if top.any() else torch.tensor(0.0, device=self.device)) +
                0.3 * boundary_height_loss
            )
        if self.enable_initial_condition_loss:
            t_norm = physics_data['t_norm']
            rx = physics_data.get('release_mask', torch.zeros_like(h))
            t0 = (t_norm < 0.01)
            if t0.any():
                non_release = (rx < 0.5) & t0
                init_h_loss = torch.mean((torch.relu(physics_data['h_phys'][non_release])).pow(2)) if non_release.any() else torch.tensor(0.0, device=self.device)
                vthr = self.initial_velocity_threshold
                vx = physics_data['vx_phys']; vy = physics_data['vy_phys']
                v_excess = torch.relu(torch.abs(vx[t0]) - vthr).pow(2).mean() + torch.relu(torch.abs(vy[t0]) - vthr).pow(2).mean()
                initial_loss = 0.1 * init_h_loss + 0.5 * v_excess
        total_physics_loss = (
            self.continuity_weight * continuity_loss +
            self.momentum_weight * (momentum_x_loss + momentum_y_loss) +
            self.height_constraint_weight * height_constraint +
            self.boundary_constraint_weight * boundary_loss +
            self.initial_constraint_weight * initial_loss
        )
        total_physics_loss = torch.clamp(total_physics_loss, min=self.loss_clamp_min, max=self.loss_clamp_max)
        if torch.isnan(total_physics_loss) or torch.isinf(total_physics_loss):
            logging.warning("Physics loss is NaN/Inf, returning zero")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        try:
            self.last_component_losses = {
                : float(continuity_loss.item()),
                : float(momentum_x_loss.item()),
                : float(momentum_y_loss.item()),
                : float(height_constraint.item()),
                : float(boundary_loss.item()),
                : float(initial_loss.item())
            }
        except Exception:
            pass
        return total_physics_loss
    def compute_active_region_mask(self, h_pred: torch.Tensor, vx_pred: torch.Tensor, vy_pred: torch.Tensor) -> torch.Tensor:
        if len(h_pred.shape) == 4:
            h_pred = h_pred.squeeze(1)  
        active_mask = (h_pred > self.active_height_threshold).float()
        return active_mask
    def compute_physics_loss(self, x_batch, pred_norm, input_norm_stats, target_norm_stats, step=None, 
                           active_sampling_rate=None, boundary_weight=None, model=None,
                           static_data=None, physics_params=None):
        try:
            return self.compute_dimensionless_physics_loss(
                x_batch=x_batch,
                predictions=pred_norm if pred_norm is not None else model(x_batch),
                targets=None,  
                model=model,  
                active_sampling_rate=active_sampling_rate,
                static_data=static_data,
                physics_params=physics_params
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
                negative_height_penalty = torch.mean(torch.relu(-h)**2 * initial_mask)
                excessive_height_penalty = torch.mean(torch.relu(h - self.height_max_limit)**2 * initial_mask)
                initial_loss = (
                    1.0 * init_velocity_loss +
                    0.5 * release_height_loss +
                    0.3 * non_release_height_loss +
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
AvalanchePhysicsLoss = DimensionlessPhysicsLoss
create_physics_loss = create_dimensionless_physics_loss
if False:
    def compute_residual_loss_on_points(
        self,
        states: Dict[str, torch.Tensor],
        grads: Dict[str, torch.Tensor],
        static: Dict[str, torch.Tensor],
        active_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        try:
            h = states['h']; vx = states['vx']; vy = states['vy']
            dh_dt = grads['dh_dt']; dh_dx = grads['dh_dx']; dh_dy = grads['dh_dy']
            dvx_dt = grads['dvx_dt']; dvx_dx = grads['dvx_dx']; dvx_dy = grads['dvx_dy']
            dvy_dt = grads['dvy_dt']; dvy_dx = grads['dvy_dx']; dvy_dy = grads['dvy_dy']
            dzdx = static['dzdx']; dzdy = static['dzdy']
            mu_phys = static.get('mu', torch.zeros_like(h))
            xi_phys = static.get('xi', torch.ones_like(h))
            rho_phys = static.get('rho', torch.full_like(h, 200.0))
            dflux_x_dx = dh_dx * vx + h * dvx_dx  
            dflux_y_dy = dh_dy * vy + h * dvy_dy  
            continuity_residual = dh_dt + dflux_x_dx + dflux_y_dy
            T_char = self.dx / self.U_char
            continuity_scale = self.H_char / T_char  
            continuity_dimensionless = continuity_residual / continuity_scale
            convection_x = vx * dvx_dx + vy * dvx_dy
            convection_y = vx * dvy_dx + vy * dvy_dy
            pressure_x = self.g * dh_dx
            pressure_y = self.g * dh_dy
            terrain_x = self.g * dzdx
            terrain_y = self.g * dzdy
            u_mag = torch.sqrt(vx**2 + vy**2 + self.eps)
            u_mag = torch.clamp(u_mag, min=self.eps, max=1e3)
            n_ux = vx / u_mag
            n_uy = vy / u_mag
            xi_phys_safe = torch.clamp(xi_phys, min=self.eps, max=1e6)
            friction_term = mu_phys * self.g * h + self.g * (u_mag ** 2) / xi_phys_safe
            friction_term = torch.clamp(friction_term, min=-1e6, max=1e6)
            friction_x = n_ux * friction_term
            friction_y = n_uy * friction_term
            friction_x = torch.where(torch.isnan(friction_x) | torch.isinf(friction_x), torch.zeros_like(friction_x), friction_x)
            friction_y = torch.where(torch.isnan(friction_y) | torch.isinf(friction_y), torch.zeros_like(friction_y), friction_y)
            momentum_x_residual = dvx_dt + convection_x + pressure_x + terrain_x + friction_x
            momentum_y_residual = dvy_dt + convection_y + pressure_y + terrain_y + friction_y
            momentum_x_dimensionless = momentum_x_residual / self.A_char
            momentum_y_dimensionless = momentum_y_residual / self.A_char
            if active_mask is None:
                active_mask = (h > self.active_height_threshold).float()
            active_mask = active_mask.view_as(h)
            active_points = int(torch.sum(active_mask).item())
            if active_points < 1:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            continuity_loss = torch.mean((continuity_dimensionless ** 2) * active_mask)
            momentum_x_loss = torch.mean((momentum_x_dimensionless ** 2) * active_mask)
            momentum_y_loss = torch.mean((momentum_y_dimensionless ** 2) * active_mask)
            h_dimensionless = h / self.H_char
            height_constraint = torch.mean((torch.clamp(h_dimensionless, min=0) - h_dimensionless)**2)
            total_physics_loss = (
                self.continuity_weight * continuity_loss +
                self.momentum_weight * (momentum_x_loss + momentum_y_loss) +
                self.height_constraint_weight * height_constraint
            )
            total_physics_loss = torch.clamp(total_physics_loss, min=self.loss_clamp_min, max=self.loss_clamp_max)
            return total_physics_loss
        except Exception as e:
            logging.warning(f"Point-level Physics loss calculation failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
