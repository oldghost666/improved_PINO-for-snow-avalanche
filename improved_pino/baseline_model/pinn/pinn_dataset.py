
import h5py
import numpy as np
import torch
import json
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Tuple, Optional, Any
import random
class AvalanchePINNDataset(Dataset):
    def __init__(
        self,
        h5_file_path: str,
        tile_ids: List[str],
        normalization_stats_path: str,
        global_data_config,
        num_data_points: int = 5000,
        num_physics_points: int = 3000,
        active_height_threshold: float = 0.05,
        active_sampling_ratio: float = 0.7,
        mode: str = 'train',  
        seed: int = 42,
        augment_enable: bool = False,
        time_neighbor_shift_prob: float = 0.0,
        noise_std: float = 0.0
    ):
        self.h5_file_path = Path(h5_file_path)
        self.tile_ids = tile_ids
        self.global_data_config = global_data_config
        self.num_data_points = num_data_points
        self.num_physics_points = num_physics_points
        self.active_height_threshold = active_height_threshold
        self.active_sampling_ratio = active_sampling_ratio
        self.mode = mode
        self.seed = seed
        self.augment_enable = augment_enable
        self.time_neighbor_shift_prob = float(time_neighbor_shift_prob)
        self.noise_std = float(noise_std)
        random.seed(seed)
        np.random.seed(seed)
        self.norm_stats = self._load_normalization_stats(normalization_stats_path)
        self.data_cache = {}
        self.sample_indices = []  
        self._load_data()
        self._build_sample_indices()
        logging.info(f"Avalanche PINN dataset initialization completed:")
        logging.info(f"  Mode: {mode}")
        logging.info(f"  Number of tiles: {len(tile_ids)}")
        logging.info(f"  Total samples: {len(self.sample_indices)}")
        logging.info(f"  Active region threshold: {active_height_threshold}m")
        logging.info(f"  Data points per batch: {num_data_points}")
        logging.info(f"  Physics points per batch: {num_physics_points}")
    def _load_normalization_stats(self, stats_path: str) -> Dict[str, Any]:
        try:
            with open(stats_path, 'r') as f:
                raw_stats = json.load(f)
            logging.info(f"Successfully loaded normalization stats: {stats_path}")
            return self._convert_flat_stats_to_nested(raw_stats)
        except Exception as e:
            logging.warning(f"Failed to load normalization stats {stats_path}: {e}")
            return self._get_default_normalization_stats()
    def _convert_flat_stats_to_nested(self, flat_stats: Dict[str, Any]) -> Dict[str, Any]:
        nested_stats = {}
        var_mappings = {
            : 'heights',  
            : 'velocity_x', 
            : 'velocity_y',
            : 'dem',
            : 'dzdx',
            : 'dzdy',
            : 'mu_0',
            : 'xi_0', 
            : 'rho',
            : 'cohesion'
        }
        for var_name, stats_prefix in var_mappings.items():
            var_stats = {}
            if f"{stats_prefix}_mean" in flat_stats:
                var_stats['mean'] = flat_stats[f"{stats_prefix}_mean"]
            if f"{stats_prefix}_std" in flat_stats:
                var_stats['std'] = flat_stats[f"{stats_prefix}_std"]
            if f"{stats_prefix}_min" in flat_stats:
                var_stats['min'] = flat_stats[f"{stats_prefix}_min"]
            if f"{stats_prefix}_max" in flat_stats:
                var_stats['max'] = flat_stats[f"{stats_prefix}_max"]
            if 'standard_normalized_fields' in flat_stats and stats_prefix in flat_stats['standard_normalized_fields']:
                var_stats['method'] = 'standard'
            elif 'minmax_normalized_fields' in flat_stats and stats_prefix in flat_stats['minmax_normalized_fields']:
                var_stats['method'] = 'minmax'
            else:
                if 'mean' in var_stats and 'std' in var_stats:
                    var_stats['method'] = 'standard'
                elif 'min' in var_stats and 'max' in var_stats:
                    var_stats['method'] = 'minmax'
                else:
                    var_stats['method'] = 'standard'  
            if var_stats:  
                nested_stats[var_name] = var_stats
        return nested_stats
    def _get_default_normalization_stats(self) -> Dict[str, Any]:
        return {
            : {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 50.0},
            : {'mean': 0.0, 'std': 10.0, 'min': -50.0, 'max': 50.0},
            : {'mean': 0.0, 'std': 10.0, 'min': -50.0, 'max': 50.0},
            : {'mean': 2000.0, 'std': 500.0, 'min': 1000.0, 'max': 3000.0},
            : {'mean': 0.0, 'std': 0.5, 'min': -2.0, 'max': 2.0},
            : {'mean': 0.0, 'std': 0.5, 'min': -2.0, 'max': 2.0},
            : {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0},
            : {'mean': 0.155, 'std': 0.05, 'min': 0.1, 'max': 0.2},
            : {'mean': 1000.0, 'std': 200.0, 'min': 800.0, 'max': 1200.0},
            : {'mean': 200.0, 'std': 50.0, 'min': 150.0, 'max': 250.0},
            : {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 5.0}
        }
    def _load_data(self):
        logging.info(f"Loading H5 data file: {self.h5_file_path}")
        with h5py.File(self.h5_file_path, 'r') as f:
            for tile_id in self.tile_ids:
                if tile_id not in f:
                    logging.warning(f"Tile {tile_id} does not exist in H5 file, skipping")
                    continue
                try:
                    tile_data = self._load_tile_data(f, tile_id)
                    self.data_cache[tile_id] = tile_data
                    logging.info(f"Successfully loaded tile {tile_id}: {tile_data['num_time_steps']} time steps")
                except Exception as e:
                    logging.error(f"Failed to load tile {tile_id} failed: {e}")
                    continue
        if not self.data_cache:
            raise ValueError("No tile data loaded successfully")
    def _load_tile_data(self, h5_file: h5py.File, tile_id: str) -> Dict[str, np.ndarray]:
        tile_group = h5_file[tile_id]
        dynamic_data = {
            : np.array(tile_group['heights']),      
            : np.array(tile_group['velocity_x']), 
            : np.array(tile_group['velocity_y'])  
        }
        static_data = {
            : np.array(tile_group['dem']),           
            : np.array(tile_group['dzdx']),         
            : np.array(tile_group['dzdy']),         
            : np.array(tile_group['release_mask'])  
        }
        physics_params = {}
        for param in ['mu_0', 'xi_0', 'rho', 'cohesion']:
            if param in tile_group:
                physics_params[param] = np.array(tile_group[param])
            else:
                physics_params[param] = np.array([0.155 if param == 'mu_0' else 
                                                1000.0 if param == 'xi_0' else
                                                200.0 if param == 'rho' else 0.0])
        times = np.array(tile_group['times']) if 'times' in tile_group else None
        height, width = dynamic_data['heights'].shape[1:3]
        x_coords = np.linspace(0, width * 10.0, width)  
        y_coords = np.linspace(0, height * 10.0, height)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        return {
            : dynamic_data,
            : static_data,
            : physics_params,
            : {'X': X, 'Y': Y, 'x_coords': x_coords, 'y_coords': y_coords},
            : times,
            : dynamic_data['heights'].shape[0],
            : height,
            : width
        }
    def _build_sample_indices(self):
        self.sample_indices = []
        for tile_id, tile_data in self.data_cache.items():
            num_time_steps = tile_data['num_time_steps']
            for t in range(num_time_steps - 1):
                self.sample_indices.append({
                    : tile_id,
                    : t,
                    : t + 1,
                    : len(self.sample_indices)
                })
        logging.info(f"Construction completed:{len(self.sample_indices)} time pair samples")
    def __len__(self):
        return len(self.sample_indices)
    def __getitem__(self, idx):
        sample_info = self.sample_indices[idx]
        tile_id = sample_info['tile_id']
        input_time = sample_info['input_time']
        target_time = sample_info['target_time']
        tile_data = self.data_cache[tile_id]
        input_tensor = self._build_input_tensor(tile_data, input_time)
        target_tensor = self._build_target_tensor(tile_data, target_time)
        if self.mode == 'train' and self.augment_enable:
            if self.time_neighbor_shift_prob > 0.0 and random.random() < self.time_neighbor_shift_prob:
                num_t = self.data_cache[tile_id]['num_time_steps']
                neigh = input_time + random.choice([-1, 1])
                if neigh < 0:
                    neigh = 0
                if neigh >= num_t:
                    neigh = num_t - 1
                h_nei = self._normalize_field(self.data_cache[tile_id]['dynamic']['heights'][neigh], 'height')
                vx_nei = self._normalize_field(self.data_cache[tile_id]['dynamic']['velocity_x'][neigh], 'velocity_x')
                vy_nei = self._normalize_field(self.data_cache[tile_id]['dynamic']['velocity_y'][neigh], 'velocity_y')
                alpha = 0.8 + 0.2 * random.random()  
                input_tensor[0] = torch.FloatTensor(alpha * input_tensor[0].numpy() + (1 - alpha) * h_nei)
                input_tensor[1] = torch.FloatTensor(alpha * input_tensor[1].numpy() + (1 - alpha) * vx_nei)
                input_tensor[2] = torch.FloatTensor(alpha * input_tensor[2].numpy() + (1 - alpha) * vy_nei)
            if self.noise_std > 0.0:
                noise = torch.randn_like(input_tensor[0]) * self.noise_std
                input_tensor[0] = input_tensor[0] + noise
                noise = torch.randn_like(input_tensor[1]) * self.noise_std
                input_tensor[1] = input_tensor[1] + noise
                noise = torch.randn_like(input_tensor[2]) * self.noise_std
                input_tensor[2] = input_tensor[2] + noise
        return {
            : input_tensor,      
            : target_tensor,    
            : tile_id,
            : input_time,
            : idx,
            : tile_data.get('num_time_steps', 1)
        }
    def _build_input_tensor(self, tile_data: Dict, time_step: int) -> torch.Tensor:
        height, width = tile_data['height'], tile_data['width']
        if tile_data['times'] is not None and time_step < len(tile_data['times']):
            t_physical = tile_data['times'][time_step]  
        else:
            num_time_steps = tile_data['num_time_steps']
            if num_time_steps > 1:
                dt = 200.0 / (num_time_steps - 1)  
                t_physical = time_step * dt  
            else:
                t_physical = 0.0
        t_normalized = t_physical / 200.0  
        X_norm, Y_norm = self.global_data_config.normalize_coordinates(
            tile_data['coordinates']['X'], 
            tile_data['coordinates']['Y']
        )
        h_current = self._normalize_field(
            tile_data['dynamic']['heights'][time_step], 'height')
        vx_current = self._normalize_field(
            tile_data['dynamic']['velocity_x'][time_step], 'velocity_x')
        vy_current = self._normalize_field(
            tile_data['dynamic']['velocity_y'][time_step], 'velocity_y')
        dem = self._normalize_field(tile_data['static']['dem'], 'dem')
        dzdx = self._normalize_field(tile_data['static']['dzdx'], 'dzdx')
        dzdy = self._normalize_field(tile_data['static']['dzdy'], 'dzdy')
        release_mask = tile_data['static']['release_mask'].astype(np.float32)
        physics_fields = []
        for param_name in ['mu_0', 'xi_0', 'rho', 'cohesion']:
            param_value = tile_data['physics'][param_name]
            if param_value.ndim == 0:  
                param_field = np.full((height, width), param_value, dtype=np.float32)
            elif param_value.ndim == 1 and param_value.shape[0] == 1:  
                param_field = np.full((height, width), param_value[0], dtype=np.float32)
            else:  
                param_field = param_value
            param_normalized = self._normalize_field(param_field, param_name)
            physics_fields.append(param_normalized)
        t_field = np.full((height, width), t_normalized, dtype=np.float32)
        channels = [
            h_current, vx_current, vy_current,  
            X_norm, Y_norm, t_field,              
            dem, dzdx, dzdy, release_mask,        
            *physics_fields                        
        ]
        input_tensor = np.stack(channels, axis=0)  
        return torch.FloatTensor(input_tensor)
    def _build_target_tensor(self, tile_data: Dict, time_step: int) -> torch.Tensor:
        h_next = self._normalize_field(
            tile_data['dynamic']['heights'][time_step], 'height')
        vx_next = self._normalize_field(
            tile_data['dynamic']['velocity_x'][time_step], 'velocity_x')
        vy_next = self._normalize_field(
            tile_data['dynamic']['velocity_y'][time_step], 'velocity_y')
        target_tensor = np.stack([h_next, vx_next, vy_next], axis=0)  
        return torch.FloatTensor(target_tensor)
    def _normalize_field(self, field: np.ndarray, var_name: str) -> np.ndarray:
        if var_name not in self.norm_stats:
            logging.warning(f"Not found {var_name} normalization parameters, using min-max normalization")
            return (field - field.min()) / (field.max() - field.min() + 1e-8)
        stats = self.norm_stats[var_name]
        method = stats.get('method', 'standard')
        if method == 'standard':  
            mean, std = stats['mean'], stats['std']
            return (field - mean) / (std + 1e-8)
        elif method == 'minmax':  
            min_val, max_val = stats['min'], stats['max']
            return (field - min_val) / (max_val - min_val + 1e-8)
        else:
            return field.astype(np.float32)
    def _denormalize_field(self, field: Any, var_name: str) -> Any:
        if var_name == 't' or var_name == 'time':
            return field * 200.0  
        if var_name not in self.norm_stats:
            logging.warning(f"Not found {var_name} normalization parameters, denormalization will return original values")
            return field
        stats = self.norm_stats[var_name]
        method = stats.get('method', 'standard')
        is_torch = isinstance(field, torch.Tensor)
        if is_torch:
            to_type = 'torch'
        else:
            to_type = 'numpy'
        if method == 'standard':
            mean, std = stats['mean'], stats['std']
            return field * (std + 1e-8) + mean
        elif method == 'minmax':
            min_val, max_val = stats['min'], stats['max']
            return field * (max_val - min_val + 1e-8) + min_val
        else:
            return field
    def sample_active_regions(self, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        active_samples = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.sample_indices))
            sample = self[idx]
            h_field_norm = sample['input'][0]  
            h_field_raw = self._denormalize_field(h_field_norm, 'height')
            active_mask = (h_field_raw > self.active_height_threshold)
            if active_mask.sum() > 0 and np.random.random() < self.active_sampling_ratio:
                active_samples.append(sample)
            else:
                active_samples.append(sample)
        inputs = torch.stack([s['input'] for s in active_samples])
        targets = torch.stack([s['target'] for s in active_samples])
        h_batch_norm = inputs[:, 0]  
        h_batch_raw = self._denormalize_field(h_batch_norm, 'height')
        return {
            : inputs,
            : targets,
            : h_batch_raw > self.active_height_threshold  
        }
def create_pinn_data_loaders(
    h5_file_path: str,
    normalization_stats_path: str,
    global_data_config,
    train_tiles: List[str],
    val_tiles: List[str],
    test_tiles: List[str],
    batch_size: int = 4,
    num_data_points: int = 5000,
    num_physics_points: int = 3000,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    augment_config: Optional[Dict[str, Any]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    aug = augment_config or {}
    train_dataset = AvalanchePINNDataset(
        h5_file_path=h5_file_path,
        tile_ids=train_tiles,
        normalization_stats_path=normalization_stats_path,
        global_data_config=global_data_config,
        num_data_points=num_data_points,
        num_physics_points=num_physics_points,
        mode='train',
        augment_enable=bool(aug.get('enable', False)),
        time_neighbor_shift_prob=float(aug.get('time_neighbor_shift_prob', 0.0)),
        noise_std=float(aug.get('noise_std', 0.0))
    )
    val_dataset = AvalanchePINNDataset(
        h5_file_path=h5_file_path,
        tile_ids=val_tiles,
        normalization_stats_path=normalization_stats_path,
        global_data_config=global_data_config,
        num_data_points=num_data_points // 2,
        num_physics_points=num_physics_points // 2,
        mode='val',
        augment_enable=False,
        time_neighbor_shift_prob=0.0,
        noise_std=0.0
    )
    test_dataset = AvalanchePINNDataset(
        h5_file_path=h5_file_path,
        tile_ids=test_tiles,
        normalization_stats_path=normalization_stats_path,
        global_data_config=global_data_config,
        num_data_points=num_data_points // 2,
        num_physics_points=num_physics_points // 2,
        mode='test',
        augment_enable=False,
        time_neighbor_shift_prob=0.0,
        noise_std=0.0
    )
    def _compute_sample_weights(ds: AvalanchePINNDataset) -> torch.Tensor:
        weights: List[float] = []
        t_min = 0.0
        t_max = 0.0
        for tile_id, tile_data in ds.data_cache.items():
            num_t = int(tile_data.get('num_time_steps', 1))
            if num_t > 0:
                t_max = max(t_max, float(num_t - 1))
        initial_boost = 2.0
        release_boost = 5.0
        base = 1.0
        for sample in ds.sample_indices:
            tile_id = sample['tile_id']
            t_idx = int(sample['input_time'])
            is_initial = 1.0 if t_idx == 0 else 0.0
            try:
                release_mask = ds.data_cache[tile_id]['static']['release_mask']
                rel_ratio = float(np.asarray(release_mask, dtype=np.float32).mean())
            except Exception:
                rel_ratio = 0.0
            w = base + initial_boost * is_initial + release_boost * rel_ratio
            weights.append(float(w))
        return torch.tensor(weights, dtype=torch.float)
    train_weights = _compute_sample_weights(train_dataset)
    train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    dataloader_kwargs = {
        : num_workers,
        : pin_memory and torch.cuda.is_available(),
        : prefetch_factor if num_workers > 0 else None,
        : num_workers > 0  
    }
    dataloader_kwargs = {k: v for k, v in dataloader_kwargs.items() if v is not None}
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True,
        **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs
    )
    logging.info(f"PINN data loader created:")
    logging.info(f"  Train samples: {len(train_dataset)} (batch: {len(train_loader)})")
    logging.info(f"  Val samples: {len(val_dataset)} (batch: {len(val_loader)})")
    logging.info(f"  Test samples: {len(test_dataset)} (batch: {len(test_loader)})")
    return train_loader, val_loader, test_loader
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    h5_path = "E:/PYproject/PINONN1012/PINN_Project/data/traindata812.h5"
    norm_stats_path = "E:/PYproject/PINONN1012/PINN_Project/data/normalization_stats_complete_14ch_v4.3.json"
    train_tiles = ["tile_0001", "tile_0002", "tile_0003", "tile_0004", "tile_0005"]
    val_tiles = ["tile_0006", "tile_0007"]
    test_tiles = ["tile_0008"]
    from global_data_config import GlobalDataConfig
    global_data_config = GlobalDataConfig(h5_file_path=h5_path, normalization_file_path=norm_stats_path)
    train_loader, val_loader, test_loader = create_pinn_data_loaders(
        h5_file_path=h5_path,
        normalization_stats_path=norm_stats_path,
        global_data_config=global_data_config,
        train_tiles=train_tiles,
        val_tiles=val_tiles,
        test_tiles=test_tiles,
        batch_size=2
    )
    for batch in train_loader:
        print(f"Input shape: {batch['input'].shape}")  
        print(f"Target shape: {batch['target'].shape}")  
        print(f"Active area ratio: {(batch['input'][:, 0] > 0.05).float().mean():.3f}")
        break
