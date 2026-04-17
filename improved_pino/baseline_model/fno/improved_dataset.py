
import h5py
import numpy as np
import torch
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
class ImprovedAvalancheDataset(Dataset):
    def __init__(
        self,
        h5_file_path: str,
        tile_ids: List[str],
        sequence_length: int = 1,  
        prediction_steps: int = 1,  
        normalize: bool = True,
        dx: float = 10.0,
        dy: float = 10.0,
        dt: float = 1.0,
        boundary_condition: str = 'open',  
        sliding_window_step: int = 1,  
        sampling_mode: str = 'single_step',  
    ):
        self.h5_file_path = h5_file_path
        self.tile_ids = tile_ids
        self.sequence_length = sequence_length  
        self.prediction_steps = prediction_steps  
        self.normalize = normalize
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.boundary_condition = boundary_condition  
        self.sliding_window_step = sliding_window_step  
        self.sampling_mode = sampling_mode
        self.in_dim = 6  
        self.static_dim = 4  
        self.physics_dim = 4  
        self.out_dim = 3  
        self.data_indices = []
        self.normalization_stats = {}
        self._load_data_info()
        self._build_data_indices()
        mode_name = "multisteps" if self.sampling_mode == 'multi_step' else "single-step"
        print(f"{mode_name}Avalanche dataset initialization completed:")
        print(f"  - Number of tiles: {len(self.tile_ids)}")
        print(f"  - total samples: {len(self.data_indices)}")
        print(f"  - Input channels: 6 (dynamic+coord) + 4 (static) + 4 (physics params) = 14")
        print(f"  - Output channels: 3 (h, vx, vy)")
        print(f"  - Boundary condition: {self.boundary_condition}")
        if self.sampling_mode == 'multi_step':
            print(f"  - Sampling mode: Multi-step autoregressive ({self.sequence_length}frame sliding window)")
            print(f"  - Sequence length: {self.sequence_length}frames (1 input + {self.prediction_steps} targets)")
            print(f"  - Sliding window step: {self.sliding_window_step}")
        else:
            print(f"  - Sampling mode: Single-step prediction (t_i -> t_{{i+1}})")
            print(f"  - Sequence length: 1frame input -> 1 frame output")
    def _load_data_info(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            available_tiles = [key for key in f.keys() if key != 'metadata']
            print(f"Available tiles in H5 file: {available_tiles}")
            if self.tile_ids:
                invalid_tiles = [tile for tile in self.tile_ids if tile not in available_tiles]
                if invalid_tiles:
                    raise ValueError(f"Specified tiles do not exist in H5 file: {invalid_tiles}")
                print(f"Using specified {len(self.tile_ids)} tiles: {self.tile_ids}")
            else:
                self.tile_ids = available_tiles
                print(f"Using all {len(self.tile_ids)} tiles: {self.tile_ids}")
            try:
                from global_data_config import get_global_data_config
                global_config = get_global_data_config(self.h5_file_path)
                loaded_params = global_config.normalization_params
                print(f"[Update] Loading normalization parameters from global config (version: {loaded_params.get('version', 'unknown')})")
            except Exception as e:
                print(f"[Warning]  Error: Unable to get normalization parameters from global config - {e}")
                raise RuntimeError(f"Normalization parameters must be obtained via global config chain")
            loaded_from_external = False
            try:
                self.normalization_stats = {
                    : loaded_params['heights_min'],
                    : loaded_params['heights_max'],
                    : loaded_params['heights_range'],
                    : loaded_params['heights_mean'],
                    : loaded_params['heights_std'],
                    : loaded_params['heights_min'],
                    : loaded_params['heights_max'],
                    : loaded_params['heights_range'],
                    : loaded_params['heights_mean'],
                    : loaded_params['heights_std'],
                    : loaded_params['velocity_x_mean'],
                    : loaded_params['velocity_x_std'],
                    : loaded_params['velocity_y_mean'],
                    : loaded_params['velocity_y_std'],
                    : [loaded_params['velocity_x_mean'], 
                                    loaded_params['velocity_y_mean']],
                    : [loaded_params['velocity_x_std'], 
                                   loaded_params['velocity_y_std']],
                    : loaded_params['dem_min'],
                    : loaded_params['dem_max'],
                    : loaded_params['dem_range'],
                    : loaded_params['dem_mean'],
                    : loaded_params['dem_std'],
                    : loaded_params['dzdx_mean'],
                    : loaded_params['dzdx_std'],
                    : loaded_params['dzdy_mean'],
                    : loaded_params['dzdy_std'],
                    : loaded_params['dzdx_min'],
                    : loaded_params['dzdx_max'],
                    : loaded_params['dzdy_min'],
                    : loaded_params['dzdy_max'],
                    : loaded_params['mu_0_min'],
                    : loaded_params['mu_0_max'],
                    : loaded_params['mu_0_range'],
                    : loaded_params['mu_0_mean'],
                    : loaded_params['mu_0_std'],
                    : loaded_params['xi_0_min'],
                    : loaded_params['xi_0_max'],
                    : loaded_params['xi_0_range'],
                    : loaded_params['xi_0_mean'],
                    : loaded_params['xi_0_std'],
                    : loaded_params['rho_min'],
                    : loaded_params['rho_max'],
                    : loaded_params['rho_range'],
                    : loaded_params['rho_mean'],
                    : loaded_params['rho_std'],
                    : loaded_params['cohesion_min'],
                    : loaded_params['cohesion_max'],
                    : loaded_params['cohesion_range'],
                    : loaded_params['cohesion_mean'],
                    : loaded_params['cohesion_std'],
                    : loaded_params['global_x_min'],
                    : loaded_params['global_x_max'],
                    : loaded_params['global_y_min'],
                    : loaded_params['global_y_max'],
                    : loaded_params['global_t_min'],
                    : loaded_params['global_t_max'],
                    : loaded_params['global_x_range'],
                    : loaded_params['global_y_range'],
                    : loaded_params['global_t_range'],
                    : loaded_params['global_x_min'],
                    : loaded_params['global_x_max'],
                    : loaded_params['global_y_min'],
                    : loaded_params['global_y_max'],
                    : loaded_params['global_t_min'],
                    : loaded_params['global_t_max'],
                    : loaded_params['global_x_range'],
                    : loaded_params['global_y_range'],
                    : loaded_params['global_t_range'],
                    : 'standard',  
                    : 'standard',  
                    : 'minmax',  
                    : 'minmax',  
                    : 'standard',  
                    : 'minmax',  
                    : 'minmax',  
                    : loaded_params['version'],
                    : self.h5_file_path
                }
                print(f"[OK] successfully load normalization parameters (version: {loaded_params['version']})")
                print(f"   - Height range: [{self.normalization_stats['heights_min']:.4f}, {self.normalization_stats['heights_max']:.4f}]")
                print(f"   - Velocity mean/std: x({self.normalization_stats['velocity_x_mean']:.4f}/{self.normalization_stats['velocity_x_std']:.4f}), y({self.normalization_stats['velocity_y_mean']:.4f}/{self.normalization_stats['velocity_y_std']:.4f})")
                print(f"   - DEM range: [{self.normalization_stats['dem_min']:.1f}, {self.normalization_stats['dem_max']:.1f}]")
                print(f"   - Physics parameter range: mu0[{self.normalization_stats['mu_0_min']:.3f}, {self.normalization_stats['mu_0_max']:.3f}], xi0[{self.normalization_stats['xi_0_min']:.0f}, {self.normalization_stats['xi_0_max']:.0f}], rho[{self.normalization_stats['rho_min']:.0f}, {self.normalization_stats['rho_max']:.0f}]")
                loaded_from_external = True
            except Exception as e:
                print(f"[Warning]  Error: Unable to process normalization parameters - {e}")
                raise RuntimeError(f"Normalization parameter processing failed, check global config chain integrity")
            if loaded_from_external:
                print(f"[OK] Successfully loaded normalization parameters from external file")
                return
            else:
                raise RuntimeError("Unable to load specified normalization parameters file")
    def _build_data_indices(self):
        if self.sampling_mode == 'multi_step':
            self._build_multi_step_indices()
        else:
            self._build_single_step_indices()
    def _build_single_step_indices(self):
        time_pairs = {}  
        with h5py.File(self.h5_file_path, 'r') as f:
            for tile_id in self.tile_ids:
                if tile_id in f:
                    tile_group = f[tile_id]
                    times = tile_group['times'][:]
                    num_timesteps = len(times)
                    for i in range(num_timesteps - 1):
                        input_time = times[i]
                        target_time = times[i + 1]
                        time_pair = (input_time, target_time)
                        sample_info = {
                            : tile_id,
                            : i,
                            : i + 1,
                            : input_time,
                            : target_time,
                            : time_pair,
                            : False
                        }
                        if time_pair not in time_pairs:
                            time_pairs[time_pair] = []
                        time_pairs[time_pair].append(sample_info)
        self._finalize_indices(time_pairs, "single-step")
    def _build_multi_step_indices(self):
        time_pairs = {}  
        with h5py.File(self.h5_file_path, 'r') as f:
            for tile_id in self.tile_ids:
                if tile_id in f:
                    tile_group = f[tile_id]
                    times = tile_group['times'][:]
                    num_timesteps = len(times)
                    window_size = 6
                    for start_idx in range(0, num_timesteps, self.sliding_window_step):
                        remaining_steps = num_timesteps - start_idx
                        if remaining_steps >= window_size:
                            sequence_times = times[start_idx:start_idx + window_size]
                            actual_sequence_length = window_size
                            needs_padding = False
                        else:
                            available_times = times[start_idx:]
                            last_time = available_times[-1]
                            padding_times = np.full(window_size - remaining_steps, last_time)
                            sequence_times = np.concatenate([available_times, padding_times])
                            actual_sequence_length = window_size
                            needs_padding = True
                        input_time = sequence_times[0]  
                        target_time = sequence_times[-1]  
                        time_pair = (input_time, target_time)
                        sample_info = {
                            : tile_id,
                            : start_idx,
                            : sequence_times,
                            : input_time,
                            : target_time,
                            : time_pair,
                            : actual_sequence_length,
                            : True,
                            : needs_padding,
                            : remaining_steps if needs_padding else window_size
                        }
                        if time_pair not in time_pairs:
                            time_pairs[time_pair] = []
                        time_pairs[time_pair].append(sample_info)
        self._finalize_indices(time_pairs, "multi-step (6-frame sliding window)")
    def _finalize_indices(self, time_pairs, mode_name):
        sorted_time_pairs = sorted(time_pairs.keys(), key=lambda x: x[0])
        self.data_indices = []
        self.time_pair_groups = []  
        for time_pair in sorted_time_pairs:
            samples = time_pairs[time_pair]
            self.time_pair_groups.append({
                : time_pair,
                : time_pair[0],
                : time_pair[1],
                : samples,
                : len(self.data_indices),
                : len(self.data_indices) + len(samples)
            })
            self.data_indices.extend(samples)
        print(f"  - Data index construction completed ({mode_name}Mode):")
        print(f"    total samples: {len(self.data_indices)}")
        print(f"    Number of time pairs: {len(self.time_pair_groups)}")
        if len(sorted_time_pairs) > 0:
            print(f"    time range: {sorted_time_pairs[0][0]:.1f}s -> {sorted_time_pairs[-1][1]:.1f}s")
            print(f"    Samples per time pair: {[len(group['samples']) for group in self.time_pair_groups[:5]]}...")
    def __len__(self):
        return len(self.data_indices)
    def __getitem__(self, idx):
        sample_info = self.data_indices[idx]
        if self.sampling_mode == 'multi_step':
            return self._get_multi_step_sample(sample_info, idx)
        else:
            return self._get_single_step_sample(sample_info, idx)
    def _get_single_step_sample(self, sample_info, idx):
        tile_id = sample_info['tile_id']
        input_time_idx = sample_info['input_time_idx']
        target_time_idx = sample_info['target_time_idx']
        with h5py.File(self.h5_file_path, 'r') as f:
            tile_group = f[tile_id]
            heights = tile_group['heights'][:]  
            velocity_x = tile_group['velocity_x'][:]  
            velocity_y = tile_group['velocity_y'][:]  
            dem = tile_group['dem'][:]  
            dzdx = tile_group['dzdx'][:]  
            dzdy = tile_group['dzdy'][:]  
            release_mask = tile_group['release_mask'][:]  
            times = tile_group['times'][:]  
            physical_params = {
                : tile_group.attrs['mu_0'],
                : tile_group.attrs['xi_0'], 
                : tile_group.attrs['rho'],
                : tile_group.attrs['cohesion'],
                : tile_group.attrs.get('g', 9.81)
            }
        H, W = dem.shape
        h_input = heights[input_time_idx]  
        vx_input = velocity_x[input_time_idx]  
        vy_input = velocity_y[input_time_idx]  
        h_target = heights[target_time_idx]  
        vx_target = velocity_x[target_time_idx]  
        vy_target = velocity_y[target_time_idx]  
        if 'grid_x' in tile_group and 'grid_y' in tile_group:
            grid_x_phys = tile_group['grid_x'][:]
            grid_y_phys = tile_group['grid_y'][:]
            if hasattr(self, 'normalization_stats'):
                x_min = self.normalization_stats['x_min']
                x_max = self.normalization_stats['x_max']
                y_min = self.normalization_stats['y_min']
                y_max = self.normalization_stats['y_max']
            else:
                x_min, x_max = 537383.51, 541333.51
                y_min, y_max = 3173732.9, 3177682.9
            X = (grid_x_phys - x_min) / (x_max - x_min)
            Y = (grid_y_phys - y_min) / (y_max - y_min)
        else:
            x_coords = np.linspace(0, 1, W)
            y_coords = np.linspace(0, 1, H)
            X, Y = np.meshgrid(x_coords, y_coords)
        t_min = self.normalization_stats['time_min']
        t_max = self.normalization_stats['time_max']
        t_norm = (times[input_time_idx] - t_min) / (t_max - t_min)
        T = np.full((H, W), t_norm)
        if self.normalize:
            height_mean = self.normalization_stats['heights_mean']
            height_std = self.normalization_stats['heights_std']
            h_input = (h_input - height_mean) / (height_std + 1e-8)
            vx_input = (vx_input - self.normalization_stats['velocity_x_mean']) / (self.normalization_stats['velocity_x_std'] + 1e-8)
            vy_input = (vy_input - self.normalization_stats['velocity_y_mean']) / (self.normalization_stats['velocity_y_std'] + 1e-8)
            h_target = (h_target - height_mean) / (height_std + 1e-8)
            vx_target = (vx_target - self.normalization_stats['velocity_x_mean']) / (self.normalization_stats['velocity_x_std'] + 1e-8)
            vy_target = (vy_target - self.normalization_stats['velocity_y_mean']) / (self.normalization_stats['velocity_y_std'] + 1e-8)
        input_dynamic = np.stack([
            h_input,   
            vx_input,  
            vy_input,  
            X,         
            Y,         
            T          
        ], axis=0)  
        if self.normalize:
            if self.normalization_stats.get('dem_normalization', 'standard') == 'minmax':
                dem_min = self.normalization_stats['dem_min']
                dem_max = self.normalization_stats['dem_max']
                dem_range = dem_max - dem_min
                if dem_range > 1e-8:
                    dem = (dem - dem_min) / dem_range
                else:
                    dem = np.zeros_like(dem)
                dem = np.clip(dem, 0.0, 1.0)
            if self.normalization_stats.get('gradient_normalization', 'standard') == 'standard':
                dzdx_mean = self.normalization_stats['dzdx_mean']
                dzdx_std = self.normalization_stats['dzdx_std']
                dzdx = (dzdx - dzdx_mean) / (dzdx_std + 1e-8)
                dzdy_mean = self.normalization_stats['dzdy_mean']
                dzdy_std = self.normalization_stats['dzdy_std']
                dzdy = (dzdy - dzdy_mean) / (dzdy_std + 1e-8)
        input_static = np.stack([
            dem,
            dzdx,
            dzdy,
            release_mask
        ], axis=0)  
        if self.normalize:
            mu_0_min = self.normalization_stats['mu_0_min']
            mu_0_max = self.normalization_stats['mu_0_max']
            mu_0_range = mu_0_max - mu_0_min
            mu_0_norm = (physical_params['mu_0'] - mu_0_min) / (mu_0_range + 1e-8)
            mu_0_norm = np.clip(mu_0_norm, 0.0, 1.0)
            xi_0_min = self.normalization_stats['xi_0_min']
            xi_0_max = self.normalization_stats['xi_0_max']
            xi_0_range = xi_0_max - xi_0_min
            xi_0_norm = (physical_params['xi_0'] - xi_0_min) / (xi_0_range + 1e-8)
            xi_0_norm = np.clip(xi_0_norm, 0.0, 1.0)
            rho_min = self.normalization_stats['rho_min']
            rho_max = self.normalization_stats['rho_max']
            rho_range = rho_max - rho_min
            rho_norm = (physical_params['rho'] - rho_min) / (rho_range + 1e-8)
            rho_norm = np.clip(rho_norm, 0.0, 1.0)
            cohesion_min = self.normalization_stats['cohesion_min']
            cohesion_max = self.normalization_stats['cohesion_max']
            cohesion_range = cohesion_max - cohesion_min
            cohesion_norm = (physical_params['cohesion'] - cohesion_min) / (cohesion_range + 1e-8)
            cohesion_norm = np.clip(cohesion_norm, 0.0, 1.0)
        else:
            mu_0_norm = physical_params['mu_0']
            xi_0_norm = physical_params['xi_0']
            rho_norm = physical_params['rho']
            cohesion_norm = physical_params['cohesion']
        input_physics = np.stack([
            np.full((H, W), mu_0_norm),
            np.full((H, W), xi_0_norm),
            np.full((H, W), rho_norm),
            np.full((H, W), cohesion_norm)
        ], axis=0)  
        target = np.stack([
            h_target,   
            vx_target,  
            vy_target   
        ], axis=0)  
        input_dynamic = torch.from_numpy(input_dynamic).float()
        input_static = torch.from_numpy(input_static).float()
        input_physics = torch.from_numpy(input_physics).float()
        target = torch.from_numpy(target).float()
        return {
            : input_dynamic,      
            : input_static,        
            : input_physics,      
            : target,                    
            : tile_id,                  
            : times[input_time_idx], 
            : times[target_time_idx], 
            : (times[input_time_idx], times[target_time_idx]) 
        }
    def _get_multi_step_sample(self, sample_info, idx):
        tile_id = sample_info['tile_id']
        start_time_idx = sample_info['start_time_idx']
        sequence_times = sample_info['sequence_times']
        sequence_length = sample_info['sequence_length']
        needs_padding = sample_info.get('needs_padding', False)
        original_length = sample_info.get('original_length', sequence_length)
        with h5py.File(self.h5_file_path, 'r') as f:
            tile_group = f[tile_id]
            heights = tile_group['heights'][:]  
            velocity_x = tile_group['velocity_x'][:]  
            velocity_y = tile_group['velocity_y'][:]  
            dem = tile_group['dem'][:]  
            dzdx = tile_group['dzdx'][:]  
            dzdy = tile_group['dzdy'][:]  
            release_mask = tile_group['release_mask'][:]  
            times = tile_group['times'][:]  
            physical_params = {
                : tile_group.attrs['mu_0'],
                : tile_group.attrs['xi_0'], 
                : tile_group.attrs['rho'],
                : tile_group.attrs['cohesion'],
                : tile_group.attrs.get('g', 9.81)
            }
        H, W = dem.shape
        if needs_padding and original_length < 6:
            h_original = heights[start_time_idx:start_time_idx + original_length]  
            vx_original = velocity_x[start_time_idx:start_time_idx + original_length]  
            vy_original = velocity_y[start_time_idx:start_time_idx + original_length]  
            last_h = h_original[-1:]  
            last_vx = vx_original[-1:]  
            last_vy = vy_original[-1:]  
            pad_frames = 6 - original_length
            h_seq = np.concatenate([h_original] + [last_h] * pad_frames, axis=0)  
            vx_seq = np.concatenate([vx_original] + [last_vx] * pad_frames, axis=0)  
            vy_seq = np.concatenate([vy_original] + [last_vy] * pad_frames, axis=0)  
        else:
            h_seq = heights[start_time_idx:start_time_idx + sequence_length]  
            vx_seq = velocity_x[start_time_idx:start_time_idx + sequence_length]  
            vy_seq = velocity_y[start_time_idx:start_time_idx + sequence_length]  
        h_current = h_seq[0]  
        vx_current = vx_seq[0]  
        vy_current = vy_seq[0]  
        if 'grid_x' in tile_group and 'grid_y' in tile_group:
            grid_x_phys = tile_group['grid_x'][:]
            grid_y_phys = tile_group['grid_y'][:]
            if hasattr(self, 'normalization_stats'):
                x_min = self.normalization_stats['x_min']
                x_max = self.normalization_stats['x_max']
                y_min = self.normalization_stats['y_min']
                y_max = self.normalization_stats['y_max']
            else:
                x_min, x_max = 537383.51, 541333.51
                y_min, y_max = 3173732.9, 3177682.9
            X = (grid_x_phys - x_min) / (x_max - x_min)
            Y = (grid_y_phys - y_min) / (y_max - y_min)
        else:
            x_coords = np.linspace(0, 1, W)
            y_coords = np.linspace(0, 1, H)
            X, Y = np.meshgrid(x_coords, y_coords)
        t_min = self.normalization_stats['time_min']
        t_max = self.normalization_stats['time_max']
        t_norm = (sequence_times[0] - t_min) / (t_max - t_min)  
        T = np.full((H, W), t_norm)
        if self.normalize:
            height_mean = self.normalization_stats['heights_mean']
            height_std = self.normalization_stats['heights_std']
            h_current = (h_current - height_mean) / (height_std + 1e-8)
            vx_current = (vx_current - self.normalization_stats['velocity_x_mean']) / (self.normalization_stats['velocity_x_std'] + 1e-8)
            vy_current = (vy_current - self.normalization_stats['velocity_y_mean']) / (self.normalization_stats['velocity_y_std'] + 1e-8)
            h_seq_norm = (h_seq - height_mean) / (height_std + 1e-8)
            vx_seq_norm = (vx_seq - self.normalization_stats['velocity_x_mean']) / (self.normalization_stats['velocity_x_std'] + 1e-8)
            vy_seq_norm = (vy_seq - self.normalization_stats['velocity_y_mean']) / (self.normalization_stats['velocity_y_std'] + 1e-8)
        else:
            h_seq_norm = h_seq
            vx_seq_norm = vx_seq
            vy_seq_norm = vy_seq
        input_dynamic = np.stack([
            h_current,   
            vx_current,  
            vy_current,  
            X,           
            Y,           
            T            
        ], axis=0)  
        if self.normalize:
            if self.normalization_stats.get('dem_normalization', 'standard') == 'minmax':
                dem_min = self.normalization_stats['dem_min']
                dem_max = self.normalization_stats['dem_max']
                dem_range = dem_max - dem_min
                if dem_range > 1e-8:
                    dem = (dem - dem_min) / dem_range
                else:
                    dem = np.zeros_like(dem)
                dem = np.clip(dem, 0.0, 1.0)
            if self.normalization_stats.get('gradient_normalization', 'standard') == 'standard':
                dzdx_mean = self.normalization_stats['dzdx_mean']
                dzdx_std = self.normalization_stats['dzdx_std']
                dzdx = (dzdx - dzdx_mean) / (dzdx_std + 1e-8)
                dzdy_mean = self.normalization_stats['dzdy_mean']
                dzdy_std = self.normalization_stats['dzdy_std']
                dzdy = (dzdy - dzdy_mean) / (dzdy_std + 1e-8)
            elif self.normalization_stats.get('gradient_normalization', 'standard') == 'minmax':
                dzdx_min = self.normalization_stats['dzdx_min']
                dzdx_max = self.normalization_stats['dzdx_max']
                dzdx_range = dzdx_max - dzdx_min
                if dzdx_range > 1e-8:
                    dzdx = (dzdx - dzdx_min) / dzdx_range
                else:
                    dzdx = np.zeros_like(dzdx)
                dzdx = np.clip(dzdx, 0.0, 1.0)
                dzdy_min = self.normalization_stats['dzdy_min']
                dzdy_max = self.normalization_stats['dzdy_max']
                dzdy_range = dzdy_max - dzdy_min
                if dzdy_range > 1e-8:
                    dzdy = (dzdy - dzdy_min) / dzdy_range
                else:
                    dzdy = np.zeros_like(dzdy)
                dzdy = np.clip(dzdy, 0.0, 1.0)
        input_static = np.stack([
            dem,
            dzdx,
            dzdy,
            release_mask
        ], axis=0)  
        if self.normalize:
            mu_0_min = self.normalization_stats['mu_0_min']
            mu_0_max = self.normalization_stats['mu_0_max']
            mu_0_range = mu_0_max - mu_0_min
            mu_0_norm = (physical_params['mu_0'] - mu_0_min) / (mu_0_range + 1e-8)
            mu_0_norm = np.clip(mu_0_norm, 0.0, 1.0)
            xi_0_min = self.normalization_stats['xi_0_min']
            xi_0_max = self.normalization_stats['xi_0_max']
            xi_0_range = xi_0_max - xi_0_min
            xi_0_norm = (physical_params['xi_0'] - xi_0_min) / (xi_0_range + 1e-8)
            xi_0_norm = np.clip(xi_0_norm, 0.0, 1.0)
            rho_min = self.normalization_stats['rho_min']
            rho_max = self.normalization_stats['rho_max']
            rho_range = rho_max - rho_min
            rho_norm = (physical_params['rho'] - rho_min) / (rho_range + 1e-8)
            rho_norm = np.clip(rho_norm, 0.0, 1.0)
            cohesion_min = self.normalization_stats['cohesion_min']
            cohesion_max = self.normalization_stats['cohesion_max']
            cohesion_range = cohesion_max - cohesion_min
            cohesion_norm = (physical_params['cohesion'] - cohesion_min) / (cohesion_range + 1e-8)
            cohesion_norm = np.clip(cohesion_norm, 0.0, 1.0)
        else:
            mu_0_norm = physical_params['mu_0']
            xi_0_norm = physical_params['xi_0']
            rho_norm = physical_params['rho']
            cohesion_norm = physical_params['cohesion']
        input_physics = np.stack([
            np.full((H, W), mu_0_norm),
            np.full((H, W), xi_0_norm),
            np.full((H, W), rho_norm),
            np.full((H, W), cohesion_norm)
        ], axis=0)  
        target_seq = []
        for step in range(1, 6):  
            target_step = np.stack([
                h_seq_norm[step],   
                vx_seq_norm[step],  
                vy_seq_norm[step]   
            ], axis=0)  
            target_seq.append(target_step)
        target_seq = np.stack(target_seq, axis=0)  
        input_dynamic = torch.from_numpy(input_dynamic).float()
        input_static = torch.from_numpy(input_static).float()
        input_physics = torch.from_numpy(input_physics).float()
        target_seq = torch.from_numpy(target_seq).float()
        return {
            : input_dynamic,      
            : input_static,      
            : input_physics,    
            : target_seq,          
            : tile_id,                
            : sequence_times[0],   
            : sequence_times[5] if len(sequence_times) > 5 else sequence_times[-1], 
            : (sequence_times[0], sequence_times[-1]), 
            : sequence_times,   
            : needs_padding,   
            : original_length 
        }
    def denormalize_height(self, normalized_height):
        height_mean = self.normalization_stats['heights_mean']
        height_std = self.normalization_stats['heights_std']
        if not hasattr(self, '_denorm_print_count'):
            self._denorm_print_count = 0
        if isinstance(normalized_height, np.ndarray) and normalized_height.size > 0 and self._denorm_print_count % 500 == 0:
            print(f"[Denormalization{self._denorm_print_count}] Height denormalization - normalized range: [{np.min(normalized_height):.4f}, {np.max(normalized_height):.4f}]")
            print(f"[Denormalization{self._denorm_print_count}] Height denormalization - using parameters: mean={height_mean:.4f}, std={height_std:.4f}")
        self._denorm_print_count += 1
        true_height_max = self.normalization_stats['heights_max']  
        true_height_min = self.normalization_stats['heights_min']  
        norm_max = (true_height_max - height_mean) / height_std  
        norm_min = (true_height_min - height_mean) / height_std  
        clip_min = norm_min - 2.0  
        clip_max = norm_max + 2.0  
        if isinstance(normalized_height, np.ndarray):
            normalized_height_clipped = np.clip(normalized_height, clip_min, clip_max)
        else:
            normalized_height_clipped = np.clip(normalized_height, clip_min, clip_max)
        denormalized = normalized_height_clipped * height_std + height_mean
        if isinstance(denormalized, np.ndarray):
            denormalized = np.maximum(denormalized, -0.1)
        else:
            denormalized = max(denormalized, -0.1)
        return denormalized
    def denormalize_coordinates(self, normalized_x, normalized_y):
        x_min = self.normalization_stats['global_x_min']
        x_max = self.normalization_stats['global_x_max']
        y_min = self.normalization_stats['global_y_min']
        y_max = self.normalization_stats['global_y_max']
        x_orig = normalized_x * (x_max - x_min) + x_min
        y_orig = normalized_y * (y_max - y_min) + y_min
        return x_orig, y_orig
    def denormalize_time(self, normalized_time):
        t_min = self.normalization_stats['global_t_min']
        t_max = self.normalization_stats['global_t_max']
        return normalized_time * (t_max - t_min) + t_min
    def denormalize_velocity(self, normalized_velocity_x, normalized_velocity_y):
        velocity_x_std = self.normalization_stats['velocity_x_std']
        velocity_x_mean = self.normalization_stats['velocity_x_mean']
        velocity_y_std = self.normalization_stats['velocity_y_std']
        velocity_y_mean = self.normalization_stats['velocity_y_mean']
        if not hasattr(self, '_vel_denorm_print_count'):
            self._vel_denorm_print_count = 0
        if isinstance(normalized_velocity_x, np.ndarray) and normalized_velocity_x.size > 0 and self._vel_denorm_print_count % 500 == 0:
            print(f"[Velocity denormalization{self._vel_denorm_print_count}] vx normalized range: [{np.min(normalized_velocity_x):.4f}, {np.max(normalized_velocity_x):.4f}]")
            print(f"[Velocity denormalization{self._vel_denorm_print_count}] vy normalized range: [{np.min(normalized_velocity_y):.4f}, {np.max(normalized_velocity_y):.4f}]")
            print(f"[Velocity denormalization{self._vel_denorm_print_count}] using parameters: vx_std={velocity_x_std:.4f}, vx_mean={velocity_x_mean:.4f}, vy_std={velocity_y_std:.4f}, vy_mean={velocity_y_mean:.4f}")
        self._vel_denorm_print_count += 1
        vx_min_real = self.normalization_stats.get('velocity_x_min', -75.0)
        vx_max_real = self.normalization_stats.get('velocity_x_max', 70.0)
        vy_min_real = self.normalization_stats.get('velocity_y_min', -76.0)
        vy_max_real = self.normalization_stats.get('velocity_y_max', 70.0)
        vx_min_norm = (vx_min_real - velocity_x_mean) / velocity_x_std
        vx_max_norm = (vx_max_real - velocity_x_mean) / velocity_x_std
        vy_min_norm = (vy_min_real - velocity_y_mean) / velocity_y_std
        vy_max_norm = (vy_max_real - velocity_y_mean) / velocity_y_std
        vx_range = vx_max_norm - vx_min_norm
        vy_range = vy_max_norm - vy_min_norm
        safety_margin = 0.2
        clip_min = min(vx_min_norm - vx_range * safety_margin, vy_min_norm - vy_range * safety_margin)
        clip_max = max(vx_max_norm + vx_range * safety_margin, vy_max_norm + vy_range * safety_margin)
        if isinstance(normalized_velocity_x, np.ndarray):
            normalized_velocity_x_clipped = np.clip(normalized_velocity_x, clip_min, clip_max)
            normalized_velocity_y_clipped = np.clip(normalized_velocity_y, clip_min, clip_max)
        else:
            normalized_velocity_x_clipped = np.clip(normalized_velocity_x, clip_min, clip_max)
            normalized_velocity_y_clipped = np.clip(normalized_velocity_y, clip_min, clip_max)
        vx_orig = normalized_velocity_x_clipped * velocity_x_std + velocity_x_mean
        vy_orig = normalized_velocity_y_clipped * velocity_y_std + velocity_y_mean
        return vx_orig, vy_orig
    def denormalize_physics_params(self, normalized_physics):
        if not self.normalize:
            return normalized_physics
        if isinstance(normalized_physics, dict):
            denormalized = {}
            denormalized['mu_0'] = normalized_physics.get('mu_0', 0.0) * (self.normalization_stats['mu_0_max'] - self.normalization_stats['mu_0_min']) + self.normalization_stats['mu_0_min']
            denormalized['xi_0'] = normalized_physics.get('xi_0', 0.0) * (self.normalization_stats['xi_0_max'] - self.normalization_stats['xi_0_min']) + self.normalization_stats['xi_0_min']
            denormalized['rho'] = normalized_physics.get('rho', 0.0) * (self.normalization_stats['rho_max'] - self.normalization_stats['rho_min']) + self.normalization_stats['rho_min']
            denormalized['cohesion'] = normalized_physics.get('cohesion', 0.0) * (self.normalization_stats['cohesion_max'] - self.normalization_stats['cohesion_min']) + self.normalization_stats['cohesion_min']
            return denormalized
        else:
            if isinstance(normalized_physics, torch.Tensor):
                normalized_physics = normalized_physics.detach().cpu().numpy()
            denormalized = np.zeros_like(normalized_physics)
            denormalized[0] = normalized_physics[0] * (self.normalization_stats['mu_0_max'] - self.normalization_stats['mu_0_min']) + self.normalization_stats['mu_0_min']
            denormalized[1] = normalized_physics[1] * (self.normalization_stats['xi_0_max'] - self.normalization_stats['xi_0_min']) + self.normalization_stats['xi_0_min']
            denormalized[2] = normalized_physics[2] * (self.normalization_stats['rho_max'] - self.normalization_stats['rho_min']) + self.normalization_stats['rho_min']
            denormalized[3] = normalized_physics[3] * (self.normalization_stats['cohesion_max'] - self.normalization_stats['cohesion_min']) + self.normalization_stats['cohesion_min']
            return denormalized
    def denormalize_output(self, normalized_output):
        if isinstance(normalized_output, torch.Tensor):
            normalized_output = normalized_output.detach().cpu().numpy()
        height_norm = normalized_output[0]  
        vx_norm = normalized_output[1]      
        vy_norm = normalized_output[2]      
        height_orig = self.denormalize_height(height_norm)
        vx_orig, vy_orig = self.denormalize_velocity(vx_norm, vy_norm)
        output_orig = np.stack([height_orig, vx_orig, vy_orig], axis=0)
        return output_orig
    def get_scaler_params(self):
        return {
            : self.normalization_stats,
            : self.normalize,
            : self.boundary_condition
        }
    def get_normalization_info(self):
        info = {
            : self.normalize,
            : self.boundary_condition,
            : {
                : 'standard',  
                : self.normalization_stats['velocity_normalization'],
                : self.normalization_stats['physics_normalization']
            },
            : self.normalization_stats['version']
        }
        if self.normalize and self.normalization_stats:
            if self.normalization_stats['height_normalization'] == 'minmax':
                info['height_stats'] = {
                    : self.normalization_stats['heights_min'],
                    : self.normalization_stats['heights_max'],
                    : self.normalization_stats['heights_range']
                }
            else:
                info['height_stats'] = {
                    : self.normalization_stats['heights_mean'],
                    : self.normalization_stats['heights_std']
                }
            info['velocity_stats'] = {
                : self.normalization_stats['velocity_x_mean'],
                : self.normalization_stats['velocity_x_std'],
                : self.normalization_stats['velocity_y_mean'],
                : self.normalization_stats['velocity_y_std']
            }
            info['physics_stats'] = {
                : {
                    : self.normalization_stats['mu_0_min'],
                    : self.normalization_stats['mu_0_max']
                },
                : {
                    : self.normalization_stats['xi_0_min'],
                    : self.normalization_stats['xi_0_max']
                },
                : {
                    : self.normalization_stats['rho_min'],
                    : self.normalization_stats['rho_max']
                },
                : {
                    : self.normalization_stats['cohesion_min'],
                    : self.normalization_stats['cohesion_max']
                }
            }
        return info
    def get_data_statistics(self):
        stats = {
            : len(self),
            : len(self.tile_ids),
            : self.tile_ids,
            : self.sequence_length,
            : self.prediction_steps,
            : self.in_dim + self.static_dim,
            : self.out_dim,
            : self.boundary_condition,
            : 'mixed',
            : {'dx': self.dx, 'dy': self.dy},
            : {'dt': self.dt}
        }
        if hasattr(self, 'normalization_stats'):
            stats['normalization_stats'] = self.normalization_stats
        return stats
    def get_normalization_stats(self) -> Dict:
        return self.normalization_stats.copy()
    def save_normalization_stats(self, save_path: str):
        import json
        with open(save_path, 'w') as f:
            json.dump(self.normalization_stats, f, indent=2)
        print(f"Normalization stats saved to: {save_path}")
    def load_normalization_stats(self, load_path: str):
        import json
        with open(load_path, 'r') as f:
            self.normalization_stats = json.load(f)
        print(f"Normalization stats loaded from {load_path} load")
class TimePairGroupedSampler:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.time_pair_groups = dataset.time_pair_groups
        print(f"Time pair grouping sampler initialized:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Number of time pairs: {len(self.time_pair_groups)}")
        print(f"  - Shuffle: {shuffle}")
        for i, group in enumerate(self.time_pair_groups[:3]):
            print(f"  - Time pair{i+1}: {group['input_time']:.1f}s -> {group['target_time']:.1f}s, samples: {len(group['samples'])}")
    def __iter__(self):
        import random
        for group in self.time_pair_groups:
            sample_indices = list(range(group['start_idx'], group['end_idx']))
            if self.shuffle:
                random.shuffle(sample_indices)
            for i in range(0, len(sample_indices), self.batch_size):
                batch = sample_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  
                    yield batch
                elif len(batch) > 0:  
                    yield batch
    def __len__(self):
        total_batches = 0
        for group in self.time_pair_groups:
            num_samples = len(group['samples'])
            total_batches += (num_samples + self.batch_size - 1) // self.batch_size  
        return total_batches
    def get_time_pair_info(self):
        info = []
        for i, group in enumerate(self.time_pair_groups):
            info.append({
                : i + 1,
                : group['input_time'],
                : group['target_time'],
                : len(group['samples']),
                : (len(group['samples']) + self.batch_size - 1) // self.batch_size,
                : list(set([sample['tile_id'] for sample in group['samples']]))
            })
        return info
def create_improved_dataloader(
    dataset: ImprovedAvalancheDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    group_by_time_pair: bool = True
) -> DataLoader:
    if group_by_time_pair and hasattr(dataset, 'time_pair_groups'):
        sampler = TimePairGroupedSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"Using time pair grouping data loader:")
        print(f"  - Total batches: {len(sampler)}")
        time_pair_info = sampler.get_time_pair_info()
        for info in time_pair_info[:5]:  
            print(f"  - Batch group{info['batch_group']}: {info['input_time']:.1f}s->{info['target_time']:.1f}s, "
                  )
        if len(time_pair_info) > 5:
            print(f"  - ... and{len(time_pair_info)-5}more time pairs")
        return dataloader
    elif hasattr(dataset, 'group_by_time_scale') and dataset.group_by_time_scale:
        print("warning:group_by_time_scalealreadydeprecated,recommendedusinggroup_by_time_pair")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    else:
        print(f"Using standard data loader: batch size={batch_size}, shuffle={shuffle}")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )