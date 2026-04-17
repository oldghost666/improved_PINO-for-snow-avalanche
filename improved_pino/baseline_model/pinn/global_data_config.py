
import h5py
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
class GlobalDataConfig:
    def __init__(self, h5_file_path: str = None, normalization_file_path: Optional[str] = None):
        if h5_file_path is None:
            raise ValueError("h5_file_path parameter is required. Please specify the data file path in the configuration file.")
        self.h5_file_path = h5_file_path
        import os
        from pathlib import Path
        if normalization_file_path and Path(normalization_file_path).exists():
            self.normalization_file_path = normalization_file_path
        else:
            env_path = os.environ.get('PINO_NORMALIZATION_STATS_PATH')
            if env_path and Path(env_path).exists():
                self.normalization_file_path = env_path
            else:
                current_dir = Path(__file__).parent
                self.normalization_file_path = str(current_dir / "data" / "normalization_stats_complete_14ch_v4.3.json")
        self.normalization_params = {}
        self.data_loading_config = {
            : True,  
            : False,  
            : 0.1,  
            : 1,  
            : 1,  
            : 'single_step'  
        }
        self._load_normalization_params()
    def _load_normalization_params(self):
        try:
            with open(self.normalization_file_path, 'r', encoding='utf-8') as f:
                self.normalization_params = json.load(f)
            print(f"[OK] Successfully loaded normalization parameters file: {self.normalization_file_path}")
            print(f"[Stats] Normalization strategy (optimized based on 828.txt):")
            print(f"   - Heights (heights): Standard Normalization [OK] Core Modification")
            print(f"   - Velocity Components (velocity_x, velocity_y): Standard Normalization")
            print(f"   - Velocity Magnitudes (velocities): Standard Normalization")
            print(f"   - Gradient Values (dzdx, dzdy): Standard Normalization")
            print(f"   - Other Data (dem, physics parameters): MinMax Normalization")
            print(f"   - Thin snow region weight enhancement: 2.0x")
            print(f"   - Active region threshold reduction: 0.05m")
            required_fields = [
                , 'heights_max', 'heights_mean', 'heights_std',  
                , 'velocities_max', 'velocities_mean', 'velocities_std',
                , 'velocity_x_std', 'velocity_y_mean', 'velocity_y_std',
                , 'dzdx_std', 'dzdy_mean', 'dzdy_std',
                , 'dem_max', 'mu_0_min', 'mu_0_max', 'xi_0_min', 'xi_0_max',
                , 'rho_max', 'cohesion_min', 'cohesion_max'
            ]
            missing_fields = [field for field in required_fields if field not in self.normalization_params]
            if missing_fields:
                print(f"[Warning]  Warning: Missing the following normalization parameters: {missing_fields}")
        except Exception as e:
            print(f"[Error] Failed to load normalization parameters file: {e}")
            raise ValueError(f"Unable to load normalization parameters file: {self.normalization_file_path}")
    def normalize_height(self, height):
        h_mean = self.normalization_params['heights_mean']
        h_std = self.normalization_params['heights_std']
        return (height - h_mean) / h_std
    def denormalize_height(self, height_norm):
        h_mean = self.normalization_params['heights_mean']
        h_std = self.normalization_params['heights_std']
        return height_norm * h_std + h_mean
    def normalize_velocity_magnitude(self, velocity):
        v_mean = self.normalization_params['velocities_mean']
        v_std = self.normalization_params['velocities_std']
        return (velocity - v_mean) / v_std
    def denormalize_velocity_magnitude(self, velocity_norm):
        v_mean = self.normalization_params['velocities_mean']
        v_std = self.normalization_params['velocities_std']
        return velocity_norm * v_std + v_mean
    def normalize_velocity_components(self, vx, vy):
        vx_mean = self.normalization_params['velocity_x_mean']
        vx_std = self.normalization_params['velocity_x_std']
        vy_mean = self.normalization_params['velocity_y_mean']
        vy_std = self.normalization_params['velocity_y_std']
        vx_norm = (vx - vx_mean) / vx_std
        vy_norm = (vy - vy_mean) / vy_std
        return vx_norm, vy_norm
    def denormalize_velocity_components(self, vx_norm, vy_norm):
        vx_mean = self.normalization_params['velocity_x_mean']
        vx_std = self.normalization_params['velocity_x_std']
        vy_mean = self.normalization_params['velocity_y_mean']
        vy_std = self.normalization_params['velocity_y_std']
        vx = vx_norm * vx_std + vx_mean
        vy = vy_norm * vy_std + vy_mean
        return vx, vy
    def normalize_gradients(self, dzdx, dzdy):
        dzdx_mean = self.normalization_params['dzdx_mean']
        dzdx_std = self.normalization_params['dzdx_std']
        dzdy_mean = self.normalization_params['dzdy_mean']
        dzdy_std = self.normalization_params['dzdy_std']
        dzdx_norm = (dzdx - dzdx_mean) / dzdx_std
        dzdy_norm = (dzdy - dzdy_mean) / dzdy_std
        return dzdx_norm, dzdy_norm
    def denormalize_gradients(self, dzdx_norm, dzdy_norm):
        dzdx_mean = self.normalization_params['dzdx_mean']
        dzdx_std = self.normalization_params['dzdx_std']
        dzdy_mean = self.normalization_params['dzdy_mean']
        dzdy_std = self.normalization_params['dzdy_std']
        dzdx = dzdx_norm * dzdx_std + dzdx_mean
        dzdy = dzdy_norm * dzdy_std + dzdy_mean
        return dzdx, dzdy
    def get_gradient_denorm_params(self):
        return {
            : self.normalization_params['dzdx_mean'],
            : self.normalization_params['dzdx_std'],
            : self.normalization_params['dzdy_mean'],
            : self.normalization_params['dzdy_std']
        }
    def normalize_dem(self, dem):
        dem_min = self.normalization_params['dem_min']
        dem_max = self.normalization_params['dem_max']
        return (dem - dem_min) / (dem_max - dem_min)
    def denormalize_dem(self, dem_norm):
        dem_min = self.normalization_params['dem_min']
        dem_max = self.normalization_params['dem_max']
        return dem_norm * (dem_max - dem_min) + dem_min
    def normalize_physics_params(self, mu_0=None, xi_0=None, rho=None, cohesion=None):
        result = {}
        if mu_0 is not None:
            mu_0_min = self.normalization_params['mu_0_min']
            mu_0_max = self.normalization_params['mu_0_max']
            result['mu_0'] = (mu_0 - mu_0_min) / (mu_0_max - mu_0_min)
        if xi_0 is not None:
            xi_0_min = self.normalization_params['xi_0_min']
            xi_0_max = self.normalization_params['xi_0_max']
            result['xi_0'] = (xi_0 - xi_0_min) / (xi_0_max - xi_0_min)
        if rho is not None:
            rho_min = self.normalization_params['rho_min']
            rho_max = self.normalization_params['rho_max']
            result['rho'] = (rho - rho_min) / (rho_max - rho_min)
        if cohesion is not None:
            cohesion_min = self.normalization_params['cohesion_min']
            cohesion_max = self.normalization_params['cohesion_max']
            result['cohesion'] = (cohesion - cohesion_min) / (cohesion_max - cohesion_min)
        return result
    def denormalize_physics_params(self, mu_0_norm=None, xi_0_norm=None, rho_norm=None, cohesion_norm=None):
        result = {}
        if mu_0_norm is not None:
            mu_0_min = self.normalization_params['mu_0_min']
            mu_0_max = self.normalization_params['mu_0_max']
            result['mu_0'] = mu_0_norm * (mu_0_max - mu_0_min) + mu_0_min
        if xi_0_norm is not None:
            xi_0_min = self.normalization_params['xi_0_min']
            xi_0_max = self.normalization_params['xi_0_max']
            result['xi_0'] = xi_0_norm * (xi_0_max - xi_0_min) + xi_0_min
        if rho_norm is not None:
            rho_min = self.normalization_params['rho_min']
            rho_max = self.normalization_params['rho_max']
            result['rho'] = rho_norm * (rho_max - rho_min) + rho_min
        if cohesion_norm is not None:
            cohesion_min = self.normalization_params['cohesion_min']
            cohesion_max = self.normalization_params['cohesion_max']
            result['cohesion'] = cohesion_norm * (cohesion_max - cohesion_min) + cohesion_min
        return result
    def get_physics_denorm_params(self):
        return {
            : self.normalization_params['mu_0_min'],
            : self.normalization_params['mu_0_max'],
            : self.normalization_params['xi_0_min'],
            : self.normalization_params['xi_0_max'],
            : self.normalization_params['rho_min'],
            : self.normalization_params['rho_max'],
            : self.normalization_params['cohesion_min'],
            : self.normalization_params['cohesion_max']
        }
    def normalize_coordinates(self, x_coords, y_coords):
        x_min = self.normalization_params['global_x_min']
        x_max = self.normalization_params['global_x_max']
        y_min = self.normalization_params['global_y_min']
        y_max = self.normalization_params['global_y_max']
        x_norm = (x_coords - x_min) / (x_max - x_min)
        y_norm = (y_coords - y_min) / (y_max - y_min)
        return x_norm, y_norm
    def denormalize_coordinates(self, x_norm, y_norm):
        x_min = self.normalization_params['global_x_min']
        x_max = self.normalization_params['global_x_max']
        y_min = self.normalization_params['global_y_min']
        y_max = self.normalization_params['global_y_max']
        x_coords = x_norm * (x_max - x_min) + x_min
        y_coords = y_norm * (y_max - y_min) + y_min
        return x_coords, y_coords
    def normalize_time(self, times):
        return times / 200.0
    def denormalize_time(self, times_norm):
        return times_norm * 200.0
    def get_denormalization_params(self):
        return {
            : self.normalization_params['heights_mean'],
            : self.normalization_params['heights_std'],
            : self.normalization_params['velocity_x_mean'],
            : self.normalization_params['velocity_x_std'],
            : self.normalization_params['velocity_y_mean'],
            : self.normalization_params['velocity_y_std'],
            : self.normalization_params['dem_min'],
            : self.normalization_params['dem_max'],
            : self.normalization_params['dzdx_mean'],
            : self.normalization_params['dzdx_std'],
            : self.normalization_params['dzdy_mean'],
            : self.normalization_params['dzdy_std']
        }
    def get_height_denorm_params(self):
        h_mean = self.normalization_params['heights_mean']
        h_std = self.normalization_params['heights_std']
        return h_mean, h_std
    def get_height_physical_range(self):
        h_min = self.normalization_params['heights_min']
        h_max = self.normalization_params['heights_max']
        return h_min, h_max
    def get_height_clipping_range(self, buffer_factor=1.1):
        h_min, h_max = self.get_height_physical_range()
        clip_min = 0.0
        clip_max = h_max * buffer_factor
        return clip_min, clip_max
    def get_velocity_denorm_params(self):
        return {
            : (self.normalization_params['velocity_x_mean'], self.normalization_params['velocity_x_std']),
            : (self.normalization_params['velocity_y_mean'], self.normalization_params['velocity_y_std']),
            : self.normalization_params['velocity_x_mean'],
            : self.normalization_params['velocity_x_std'],
            : self.normalization_params['velocity_y_mean'],
            : self.normalization_params['velocity_y_std'],
            : self.normalization_params['velocity_x_min'],
            : self.normalization_params['velocity_x_max'],
            : self.normalization_params['velocity_y_min'],
            : self.normalization_params['velocity_y_max'],
            : self.normalization_params['velocities_min'],
            : self.normalization_params['velocities_max']
        }
    def get_coord_scale(self):
        x_range = self.normalization_params['global_x_max'] - self.normalization_params['global_x_min']
        y_range = self.normalization_params['global_y_max'] - self.normalization_params['global_y_min']
        return max(x_range, y_range)
    def get_time_scale(self):
        return 200.0
    def validate_with_json_stats(self, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_params = json.load(f)
            print(f"\n=== Verifying normalization parameter consistency ===")
            print(f"JSON file: {json_path}")
            key_params = ['heights_min', 'heights_max', 'velocity_x_mean', 'velocity_x_std']
            all_consistent = True
            for key in key_params:
                if key in json_params and key in self.normalization_params:
                    json_val = json_params[key]
                    config_val = self.normalization_params[key]
                    if abs(json_val - config_val) > 1e-6:
                        print(f"[Warning]  Parameter inconsistency: {key} - JSON: {json_val}, Config: {config_val}")
                        all_consistent = False
                    else:
                        print(f"[OK] Parameter consistency: {key} = {json_val}")
                else:
                    print(f"[Error] Missing parameter: {key}")
                    all_consistent = False
            if all_consistent:
                print("[OK] All key parameters verified successfully")
            else:
                print("[Warning]  Parameter inconsistency exists")
        except Exception as e:
            print(f"[Error] Verification failed: {e}")
    def get_data_loading_config(self):
        return self.data_loading_config.copy()
    def update_data_loading_config(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.data_loading_config:
                self.data_loading_config[key] = value
                print(f"[OK] Updated data loading config: {key} = {value}")
            else:
                print(f"[Warning]  Warning: Unknown data loading configuration item: {key}")
    def is_time_pair_grouping_enabled(self):
        return self.data_loading_config.get('group_by_time_pair', True)
    def is_time_scale_grouping_enabled(self):
        return self.data_loading_config.get('group_by_time_scale', False)
_global_config_instance = None
def get_global_data_config(h5_file_path: str) -> GlobalDataConfig:
    global _global_config_instance
    if _global_config_instance is None or _global_config_instance.h5_file_path != h5_file_path:
        _global_config_instance = GlobalDataConfig(h5_file_path)
    return _global_config_instance
if __name__ == "__main__":
    import os
    current_dir = Path(__file__).parent
    h5_path = str(current_dir / "data" / "traindata812.h5")
    if not Path(h5_path).exists():
        print(f"Warning: Test data file does not exist: {h5_path}")
        exit(1)
    config = GlobalDataConfig(h5_path)
    print("\n=== Testing denormalization parameter retrieval ===")
    h_min, h_max = config.get_height_denorm_params()
    print(f"Height denormalization parameters: min={h_min:.6f}, max={h_max:.6f}")
    velocity_params = config.get_velocity_denorm_params()
    print(f"Velocity denormalization parameters: {velocity_params}")
    coord_scale = config.get_coord_scale()
    time_scale = config.get_time_scale()
    print(f"Coordinate scale factor: {coord_scale:.3f}")
    print(f"Time scale factor: {time_scale:.3f}")
    json_path = str(current_dir / "data" / "normalization_stats_complete_14ch_v4.3.json")
    if Path(json_path).exists():
        config.validate_with_json_stats(json_path)
