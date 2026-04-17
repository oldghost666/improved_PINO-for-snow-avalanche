
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time
from pinn_model import AvalanchePINN
from pinn_physics_loss import AvalanchePhysicsLoss
class AvalancheEvaluator:
    def __init__(
        self,
        model: AvalanchePINN,
        physics_loss: AvalanchePhysicsLoss,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        output_dir: str = './evaluation_results'
    ):
        self.model = model.to(device)
        self.physics_loss = physics_loss
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluation_history = defaultdict(list)
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
        matplotlib.rcParams['axes.unicode_minus'] = False
        try:
            import matplotlib as mpl
            mpl.use('Agg')
        except Exception:
            pass
        logging.info(f"Avalanche PINN evaluator initialized:")
        logging.info(f"  Device: {device}")
        logging.info(f"  Output directory: {self.output_dir}")
    def compute_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        use_physical_units: bool = True
    ) -> Dict[str, float]:
        metrics = {}
        if use_physical_units:
            epsilon = 0.01  
        else:
            epsilon = 1e-6  
        if use_physical_units:
            pred_h_phys, pred_vx_phys, pred_vy_phys = self._denormalize_predictions(predictions)
            target_h_phys, target_vx_phys, target_vy_phys = self._denormalize_predictions(targets)
            pred_phys = torch.cat([pred_h_phys, pred_vx_phys, pred_vy_phys], dim=1)
            target_phys = torch.cat([target_h_phys, target_vx_phys, target_vy_phys], dim=1)
            pred_np = pred_phys.detach().cpu().numpy()
            target_np = target_phys.detach().cpu().numpy()
            mse = nn.MSELoss(reduction='none')(pred_phys, target_phys)
            mae = nn.L1Loss(reduction='none')(pred_phys, target_phys)
        else:
            pred_np = predictions.detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()
            mse = nn.MSELoss(reduction='none')(predictions, targets)
            mae = nn.L1Loss(reduction='none')(predictions, targets)
        for i, var_name in enumerate(['height', 'velocity_x', 'velocity_y']):
            pred_channel = predictions[:, i]
            target_channel = targets[:, i]
            pred_channel_np = pred_np[:, i]
            target_channel_np = target_np[:, i]
            if active_mask is not None:
                valid_mask = active_mask.squeeze(1) > 0
                valid_mask_np = valid_mask.cpu().numpy()
                if valid_mask.sum() == 0:
                    mse_channel = mse[:, i].mean().item()
                    mae_channel = mae[:, i].mean().item()
                    rmse_channel = np.sqrt(mse_channel)
                    rel_error = np.mean(2.0 * np.abs(pred_channel_np - target_channel_np) / (np.abs(target_channel_np) + np.abs(pred_channel_np) + epsilon)) * 100
                    target_mean = np.mean(target_channel_np)
                    ss_tot = np.sum((target_channel_np - target_mean) ** 2)
                    ss_res = np.sum((target_channel_np - pred_channel_np) ** 2)
                    r2 = 1 - ss_res / (ss_tot + epsilon)
                else:
                    mse_channel = mse[:, i][valid_mask].mean().item()
                    mae_channel = mae[:, i][valid_mask].mean().item()
                    rmse_channel = np.sqrt(mse_channel)
                    pred_valid = pred_channel_np[valid_mask_np]
                    target_valid = target_channel_np[valid_mask_np]
                    rel_error = np.mean(2.0 * np.abs(pred_valid - target_valid) / (np.abs(target_valid) + np.abs(pred_valid) + epsilon)) * 100
                    target_mean = np.mean(target_valid)
                    ss_tot = np.sum((target_valid - target_mean) ** 2)
                    ss_res = np.sum((target_valid - pred_valid) ** 2)
                    r2 = 1 - ss_res / (ss_tot + epsilon)
            else:
                mse_channel = mse[:, i].mean().item()
                mae_channel = mae[:, i].mean().item()
                rmse_channel = np.sqrt(mse_channel)
                rel_error = np.mean(2.0 * np.abs(pred_channel_np - target_channel_np) / (np.abs(target_channel_np) + np.abs(pred_channel_np) + epsilon)) * 100
                target_mean = np.mean(target_channel_np)
                ss_tot = np.sum((target_channel_np - target_mean) ** 2)
                ss_res = np.sum((target_channel_np - pred_channel_np) ** 2)
                r2 = 1 - ss_res / (ss_tot + epsilon)
            metrics.update({
                : mse_channel,
                : mae_channel,
                : rmse_channel,
                : rel_error,
                : r2
            })
        if active_mask is not None:
            valid_mask = active_mask.squeeze(1) > 0
            valid_mask_np = valid_mask.cpu().numpy()
            if valid_mask.sum() == 0:
                total_mse = mse.mean().item()
                total_mae = mae.mean().item()
                pred_flat = pred_np.flatten()
                target_flat = target_np.flatten()
                total_rel_error = np.mean(2.0 * np.abs(pred_flat - target_flat) / (np.abs(target_flat) + np.abs(pred_flat) + epsilon))
                target_mean = np.mean(target_flat)
                ss_tot = np.sum((target_flat - target_mean) ** 2)
                ss_res = np.sum((target_flat - pred_flat) ** 2)
                total_r2 = 1 - ss_res / (ss_tot + epsilon)
            else:
                total_mse = mse[valid_mask.unsqueeze(1).expand_as(mse)].mean().item()
                total_mae = mae[valid_mask.unsqueeze(1).expand_as(mae)].mean().item()
                pred_flat = pred_np[valid_mask_np.reshape(pred_np.shape[0], 1, pred_np.shape[2], pred_np.shape[3]).repeat(pred_np.shape[1], axis=1)]
                target_flat = target_np[valid_mask_np.reshape(target_np.shape[0], 1, target_np.shape[2], target_np.shape[3]).repeat(target_np.shape[1], axis=1)]
                total_rel_error = np.mean(2.0 * np.abs(pred_flat - target_flat) / (np.abs(target_flat) + np.abs(pred_flat) + epsilon))
                target_mean = np.mean(target_flat)
                ss_tot = np.sum((target_flat - target_mean) ** 2)
                ss_res = np.sum((target_flat - pred_flat) ** 2)
                total_r2 = 1 - ss_res / (ss_tot + epsilon)
        else:
            total_mse = mse.mean().item()
            total_mae = mae.mean().item()
            pred_flat = pred_np.flatten()
            target_flat = target_np.flatten()
            total_rel_error = np.mean(2.0 * np.abs(pred_flat - target_flat) / (np.abs(target_flat) + np.abs(pred_flat) + epsilon))
            target_mean = np.mean(target_flat)
            ss_tot = np.sum((target_flat - target_mean) ** 2)
            ss_res = np.sum((target_flat - pred_flat) ** 2)
            total_r2 = 1 - ss_res / (ss_tot + epsilon)
        metrics.update({
            : total_mse,
            : total_mae,
            : np.sqrt(total_mse),
            : total_rel_error,
            : total_r2
        })
        return metrics
    def _denormalize_predictions(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gdc = getattr(self.physics_loss, 'global_data_config', None)
        if gdc is None:
            raise RuntimeError("physics_loss.global_data_config not set, cannot denormalize")
        h_mean, h_std = gdc.get_height_denorm_params()
        vel_params = gdc.get_velocity_denorm_params()
        vx_mean = vel_params['velocity_x_mean']; vx_std = vel_params['velocity_x_std']
        vy_mean = vel_params['velocity_y_mean']; vy_std = vel_params['velocity_y_std']
        dev = predictions.device; dtp = predictions.dtype
        h_mean_t = torch.tensor(h_mean, device=dev, dtype=dtp)
        h_std_t = torch.tensor(h_std, device=dev, dtype=dtp)
        vx_mean_t = torch.tensor(vx_mean, device=dev, dtype=dtp)
        vx_std_t = torch.tensor(vx_std, device=dev, dtype=dtp)
        vy_mean_t = torch.tensor(vy_mean, device=dev, dtype=dtp)
        vy_std_t = torch.tensor(vy_std, device=dev, dtype=dtp)
        h_phys = predictions[:, 0:1] * h_std_t + h_mean_t
        vx_phys = predictions[:, 1:2] * vx_std_t + vx_mean_t
        vy_phys = predictions[:, 2:3] * vy_std_t + vy_mean_t
        return h_phys, vx_phys, vy_phys
    def single_step_evaluation(
        self, 
        test_loader: torch.utils.data.DataLoader,
        save_results: bool = True
    ) -> Dict[str, Any]:
        logging.info("Starting single-step error evaluation...")
        self.model.eval()
        all_metrics = []
        all_predictions = []
        all_targets = []
        all_inputs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                predictions = self.model(inputs)
                h_phys_pred, vx_phys_pred, vy_phys_pred = self._denormalize_predictions(predictions)
                h_phys_tgt, vx_phys_tgt, vy_phys_tgt = self._denormalize_predictions(targets)
                active_mask = self.physics_loss.compute_active_region_mask(h_phys_tgt, vx_phys_tgt, vy_phys_tgt)
                metrics = self.compute_metrics(predictions, targets, active_mask, use_physical_units=True)
                all_metrics.append(metrics)
                if save_results:
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
                    all_inputs.append(inputs.cpu())
                if batch_idx % 10 == 0:
                    logging.info(f"Single-step evaluation progress: {batch_idx + 1}/{len(test_loader)}")
        avg_metrics = self._aggregate_metrics(all_metrics)
        if save_results:
            results = {
                : avg_metrics,
                : torch.cat(all_predictions, dim=0) if all_predictions else None,
                : torch.cat(all_targets, dim=0) if all_targets else None,
                : torch.cat(all_inputs, dim=0) if all_inputs else None
            }
            self._save_evaluation_results(results, 'single_step_evaluation')
            try:
                if all_predictions and all_targets:
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    save_path = str(self.output_dir / f"single_step_visual_{ts}.png")
                    preds = all_predictions[0]
                    tgts = all_targets[0]
                    inps = all_inputs[0]
                    self.visualize_predictions(preds, tgts, inps, save_path=save_path, max_samples=2)
            except Exception as viz_err:
                logging.warning(f"Single-step evaluation visualization failed: {viz_err}")
        logging.info("Single-step error evaluation completed")
        self._log_metrics(avg_metrics, "Single-step evaluation")
        return avg_metrics
    def multi_step_rollout(
        self, 
        initial_state: torch.Tensor,
        num_steps: int = 10,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        logging.info(f"Starting multi-step rollout prediction ({num_steps} steps)...")
        self.model.eval()
        current_state = initial_state.to(self.device)
        rollout_results = {
            : [],
            : [],
            : [],
            : []
        }
        with torch.no_grad():
            for step in range(num_steps):
                predictions = self.model(current_state)
                h_phys, vx_phys, vy_phys = self._denormalize_predictions(predictions)
                mass_conservation = self._check_mass_conservation(h_phys)
                energy_conservation = self._check_energy_conservation(h_phys, vx_phys, vy_phys)
                active_mask = self.physics_loss.compute_active_region_mask(h_phys, vx_phys, vy_phys)
                active_area_ratio = active_mask.float().mean().item()
                if save_intermediate:
                    rollout_results['predictions'].append(predictions.cpu())
                rollout_results['mass_conservation'].append(mass_conservation)
                rollout_results['energy_conservation'].append(energy_conservation)
                rollout_results['active_area_ratio'].append(active_area_ratio)
                current_state = self._update_state_with_predictions(current_state, predictions)
                if step % 2 == 0:
                    logging.info(f"Rollout prediction progress: {step + 1}/{num_steps}, Active area ratio: {active_area_ratio:.3f}")
        final_results = {
            : num_steps,
            : np.mean(rollout_results['mass_conservation']),
            : np.mean(rollout_results['energy_conservation']),
            : rollout_results['active_area_ratio'][-1],
            : self._compute_drift(rollout_results['mass_conservation']),
            : self._compute_drift(rollout_results['energy_conservation'])
        }
        if save_intermediate:
            final_results['predictions'] = torch.cat(rollout_results['predictions'], dim=0)
            final_results['mass_history'] = rollout_results['mass_conservation']
            final_results['energy_history'] = rollout_results['energy_conservation']
            final_results['active_area_history'] = rollout_results['active_area_ratio']
        self._save_evaluation_results(final_results, 'multi_step_rollout')
        logging.info("Multi-step rollout prediction completed")
        self._log_rollout_results(final_results)
        return final_results
    def _check_mass_conservation(self, h: torch.Tensor) -> float:
        dx = float(getattr(self.physics_loss, 'dx', 1.0))
        dy = float(getattr(self.physics_loss, 'dy', 1.0))
        dA = dx * dy
        total_volume = (h * dA).sum(dim=[2, 3]).mean().item()
        if not hasattr(self, '_initial_mass'):
            self._initial_mass = total_volume
        mass_conservation_error = abs(total_volume - self._initial_mass) / (self._initial_mass + 1e-8)
        return mass_conservation_error
    def _check_energy_conservation(self, h: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor) -> float:
        dx = float(getattr(self.physics_loss, 'dx', 1.0))
        dy = float(getattr(self.physics_loss, 'dy', 1.0))
        dA = dx * dy
        g = float(getattr(self.physics_loss, 'g', 9.81))
        potential_energy = (0.5 * g * (h ** 2) * dA).sum(dim=[2, 3])
        kinetic_energy = (0.5 * h * (vx ** 2 + vy ** 2) * dA).sum(dim=[2, 3])
        total_energy = (potential_energy + kinetic_energy).mean().item()
        if not hasattr(self, '_initial_energy'):
            self._initial_energy = total_energy
        energy_conservation_error = abs(total_energy - self._initial_energy) / (self._initial_energy + 1e-8)
        return energy_conservation_error
    def _compute_drift(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return abs(slope)
    def _update_state_with_predictions(
        self, 
        current_state: torch.Tensor, 
        predictions: torch.Tensor
    ) -> torch.Tensor:
        dem = current_state[:, 6:7]      
        dzdx = current_state[:, 7:8]     
        dzdy = current_state[:, 8:9]     
        release_mask = current_state[:, 9:10]  
        physics_params = current_state[:, 10:14]  
        h_pred = predictions[:, 0:1]     
        vx_pred = predictions[:, 1:2]    
        vy_pred = predictions[:, 2:3]    
        batch_size, _, height, width = current_state.shape
        x_coords = torch.linspace(0, 1, width, device=current_state.device)
        y_coords = torch.linspace(0, 1, height, device=current_state.device)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        X = X.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        Y = Y.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        current_time = current_state[:, 5:6] + 0.1
        new_state = torch.cat([
            h_pred, vx_pred, vy_pred,    
            X, Y, current_time,          
            dem, dzdx, dzdy, release_mask,  
            physics_params               
        ], dim=1)
        return new_state
    def visualize_predictions(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor,
        save_path: str = None,
        max_samples: int = 4
    ):
        batch_size = min(predictions.shape[0], max_samples)
        fig, axes = plt.subplots(batch_size, 9, figsize=(18, 4 * batch_size))
        def _imshow(ax, data, title, cmap='viridis'):
            im = ax.imshow(data, cmap=cmap, aspect='auto')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(batch_size):
            h_pred = predictions[i, 0].detach().cpu().numpy()
            vx_pred = predictions[i, 1].detach().cpu().numpy()
            vy_pred = predictions[i, 2].detach().cpu().numpy()
            h_tgt = targets[i, 0].detach().cpu().numpy()
            vx_tgt = targets[i, 1].detach().cpu().numpy()
            vy_tgt = targets[i, 2].detach().cpu().numpy()
            h_err = np.abs(h_pred - h_tgt)
            vx_err = np.abs(vx_pred - vx_tgt)
            vy_err = np.abs(vy_pred - vy_tgt)
            row = axes[i] if batch_size > 1 else axes
            _imshow(row[0], h_pred, f'Sample{i+1}: Predicted H')
            _imshow(row[1], h_tgt, f'Sample{i+1}: True H')
            _imshow(row[2], h_err, f'Sample{i+1}: Error |H|', cmap='Reds')
            _imshow(row[3], vx_pred, f'Sample{i+1}: Predicted Ux')
            _imshow(row[4], vx_tgt, f'Sample{i+1}: True Ux')
            _imshow(row[5], vx_err, f'Sample{i+1}: Error |Ux|', cmap='Reds')
            _imshow(row[6], vy_pred, f'Sample{i+1}: Predicted Uy')
            _imshow(row[7], vy_tgt, f'Sample{i+1}: True Uy')
            _imshow(row[8], vy_err, f'Sample{i+1}: Error |Uy|', cmap='Reds')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Prediction visualization saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()
    def visualize_from_loader(
        self,
        loader: torch.utils.data.DataLoader,
        save_prefix: str = 'val_visual',
        max_batches: int = 1,
        max_samples: int = 2
    ):
        self.model.eval()
        batches_done = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                preds = self.model(inputs)
                ts = time.strftime('%Y%m%d_%H%M%S')
                save_path = str(self.output_dir / f"{save_prefix}_{ts}_b{batches_done}.png")
                self.visualize_predictions(preds, targets, inputs, save_path=save_path, max_samples=max_samples)
                batches_done += 1
                if batches_done >= max_batches:
                    break
    def visualize_val_sequence(
        self,
        loader: torch.utils.data.DataLoader,
        steps: int = 10,
        samples_per_step: int = 1,
        save_prefix: str = 'val_sequence'
    ):
        self.model.eval()
        frame_paths = []
        step = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                preds = self.model(inputs)
                ts = time.strftime('%Y%m%d_%H%M%S')
                frame_path = str(self.output_dir / f"{save_prefix}_{ts}_step{step:03d}.png")
                self.visualize_predictions(
                    preds[:samples_per_step], targets[:samples_per_step], inputs[:samples_per_step],
                    save_path=frame_path, max_samples=samples_per_step
                )
                frame_paths.append(frame_path)
                step += 1
                if step >= steps:
                    break
        try:
            import imageio
            images = []
            for p in frame_paths:
                images.append(imageio.imread(p))
            gif_path = str(self.output_dir / f"{save_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.gif")
            imageio.mimsave(gif_path, images, fps=2)
            logging.info(f"Sequence GIF saved to: {gif_path}")
        except Exception as _:
            pass
    def generate_evaluation_report(
        self,
        single_step_results: Dict[str, Any],
        multi_step_results: Dict[str, Any],
        save_path: str = None
    ) -> str:
        report = []
        report.append("=" * 60)
        report.append("Avalanche PINN Model Evaluation Report")
        report.append("=" * 60)
        report.append("")
        report.append("1. Single-step Error Evaluation")
        report.append("-" * 30)
        if 'metrics' in single_step_results:
            metrics = single_step_results['metrics']
            report.append(f"Total MSE: {metrics.get('total_mse', 'N/A'):.6f}")
            report.append(f"Total MAE: {metrics.get('total_mae', 'N/A'):.6f}")
            report.append(f"Total RMSE: {metrics.get('total_rmse', 'N/A'):.6f}")
            report.append("")
            report.append("Metrics per variable:")
            for var in ['height', 'velocity_x', 'velocity_y']:
                mse = metrics.get(f'{var}_mse', 'N/A')
                mae = metrics.get(f'{var}_mae', 'N/A')
                rmse = metrics.get(f'{var}_rmse', 'N/A')
                report.append(f"  {var}: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
        report.append("")
        report.append("2. Multi-step Rollout Prediction")
        report.append("-" * 30)
        if 'average_mass_conservation' in multi_step_results:
            report.append(f"Average Mass Conservation Error: {multi_step_results['average_mass_conservation']:.6f}")
            report.append(f"Average Energy Conservation Error: {multi_step_results['average_energy_conservation']:.6f}")
            report.append(f"Mass Drift: {multi_step_results.get('mass_drift', 'N/A'):.6f}")
            report.append(f"Energy Drift: {multi_step_results.get('energy_drift', 'N/A'):.6f}")
            report.append(f"Final Active Area Ratio: {multi_step_results.get('final_active_area_ratio', 'N/A'):.3f}")
        report.append("")
        report.append("3. Model Information")
        report.append("-" * 30)
        model_info = self.model.get_model_info()
        report.append(f"Total Parameters: {model_info.get('total_parameters', 'N/A'):,}")
        report.append(f"Trainable Parameters: {model_info.get('trainable_parameters', 'N/A'):,}")
        report.append(f"Hidden Dimension: {model_info.get('hidden_dim', 'N/A')}")
        report.append(f"Number of Hidden Layers: {model_info.get('num_hidden_layers', 'N/A')}")
        report.append("")
        report.append("=" * 60)
        report.append("Evaluation report generated")
        report.append("=" * 60)
        report_text = "\n".join(report)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logging.info(f"Evaluation report saved to: {save_path}")
        return report_text
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
        return aggregated
    def _save_evaluation_results(self, results: Dict[str, Any], prefix: str):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"{prefix}_{timestamp}.json"
        serializable_results = self._make_serializable(results)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Evaluation results saved to: {save_path}")
    def _make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        print(f"\n{prefix}model evaluation results:")
        print("=" * 60)
        print(f"{'Variable':<10} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'Relative Error (%)':<15} {'R2':<10}")
        print("-" * 60)
        var_mapping = {
            : 'Avalanche Height (H)',
            : 'x-velocity (Ux)', 
            : 'y-velocity (Uy)'
        }
        for var_name in ['height', 'velocity_x', 'velocity_y']:
            display_name = var_mapping.get(var_name, var_name)
            mse = metrics.get(f'{var_name}_mse', 0.0)
            rmse = metrics.get(f'{var_name}_rmse', 0.0)
            mae = metrics.get(f'{var_name}_mae', 0.0)
            rel_error = metrics.get(f'{var_name}_rel_error', 0.0) * 100  
            r2 = metrics.get(f'{var_name}_r2', 0.0)
            print(f"{display_name:<10} {mse:<12.6f} {rmse:<12.6f} {mae:<12.6f} "
                  )
        print("=" * 60)
        total_mse = metrics.get('total_mse', 0.0)
        total_rmse = metrics.get('total_rmse', 0.0)
        total_mae = metrics.get('total_mae', 0.0)
        total_rel_error = metrics.get('total_rel_error', 0.0) * 100  
        total_r2 = metrics.get('total_r2', 0.0)
        print(f"\nOverall Evaluation Metrics:")
        print(f"Total MSE: {total_mse:.6f}")
        print(f"Total RMSE: {total_rmse:.6f}")
        print(f"Total MAE: {total_mae:.6f}")
        print(f"Total Relative Error: {total_rel_error:.2f}%")
        print(f"Total R2: {total_r2:.4f}")
        logging.info(f"{prefix} Metrics:")
        for key, value in metrics.items():
            if not key.endswith('_std'):
                std_key = f"{key}_std"
                if std_key in metrics:
                    logging.info(f"  {key}: {value:.6f} ± {metrics[std_key]:.6f}")
                else:
                    logging.info(f"  {key}: {value:.6f}")
    def _log_rollout_results(self, results: Dict[str, Any]):
        logging.info("Multi-step rollout prediction results:")
        logging.info(f"  Average Mass Conservation Error: {results.get('average_mass_conservation', 'N/A'):.6f}")
        logging.info(f"  Average Energy Conservation Error: {results.get('average_energy_conservation', 'N/A'):.6f}")
        logging.info(f"  Mass Drift: {results.get('mass_drift', 'N/A'):.6f}")
        logging.info(f"  Energy Drift: {results.get('energy_drift', 'N/A'):.6f}")
        logging.info(f"  Final Active Area Ratio: {results.get('final_active_area_ratio', 'N/A'):.3f}")
def create_evaluator(
    model: AvalanchePINN,
    physics_loss: AvalanchePhysicsLoss,
    config: Dict[str, Any]
) -> AvalancheEvaluator:
    return AvalancheEvaluator(
        model=model,
        physics_loss=physics_loss,
        device=torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
        output_dir=config.get('output_dir', './evaluation_results')
    )
if __name__ == "__main__":
    import argparse
    import yaml
    from global_data_config import GlobalDataConfig
    from pinn_dataset import AvalanchePINNDataset
    from pinn_model import EnhancedAvalanchePINN
    from pinn_physics_loss import DimensionlessPhysicsLoss
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="PINN Model Evaluation Tool")
    parser.add_argument("--config", type=str, required=True, help="Config file path (yaml)")
    parser.add_argument("--model_path", type=str, required=True, help="Model weight path (.pth)")
    parser.add_argument("--test_tile_id", type=str, default="tile_0012", help="Tile ID for testing")
    parser.add_argument("--output_dir", type=str, default="results_pinn/evaluation", help="Output directory for evaluation results")
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_config(f)
    norm_file = config.get('data', {}).get('normalization_file', 'data/normalization_stats_complete_14ch_v4.3.json')
    h5_file = config.get('data', {}).get('h5_file_path', 'data/traindata812.h5')
    global_data_config = GlobalDataConfig(norm_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model_config = config.get('model', {})
    model = EnhancedAvalanchePINN(
        input_channels=model_config.get('input_channels', 14),
        output_channels=model_config.get('output_channels', 3),
        hidden_dim=model_config.get('hidden_width', 128),
        num_layers=model_config.get('hidden_layers', 8),  
        dropout_rate=model_config.get('dropout_rate', 0.1),
        use_batch_norm=model_config.get('use_batch_norm', False),
        velocity_clip_val=model_config.get('output_constraints', {}).get('velocity_clipping', 5.0)
    ).to(device)
    logging.info(f"Loading model weights: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    physics_loss = DimensionlessPhysicsLoss(config, global_data_config, device=device)
    dataset_config = config.get('data', {}).copy()
    dataset_config['test_tile_ids'] = [args.test_tile_id] 
    test_dataset = AvalanchePINNDataset(
        h5_file_path=h5_file,
        global_data_config=global_data_config,
        mode='test',
        tile_ids=[args.test_tile_id],
        active_height_threshold=config.get('data', {}).get('sampling', {}).get('active_height_threshold', 0.05),
        points_per_sample=None, 
        physics_points_per_sample=None,
        time_window_size=config.get('data', {}).get('sequence_length', 2),
        config=dataset_config
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=0
    )
    evaluator_config = {
        : str(device),
        : args.output_dir
    }
    evaluator = AvalancheEvaluator(model, physics_loss, device, args.output_dir)
    logging.info(f"Starting evaluation for tile {args.test_tile_id} evaluating...")
    metrics = evaluator.single_step_evaluation(test_loader, save_results=True)
    evaluator.visualize_val_sequence(test_loader, steps=20, save_prefix=f'eval_{args.test_tile_id}')
    logging.info("evaluation completed!")
