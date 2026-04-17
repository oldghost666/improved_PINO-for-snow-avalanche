
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import yaml
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
try:
    from matplotlib_config import setup_matplotlib_for_chinese
    setup_matplotlib_for_chinese(verbose=False)
except Exception:
    pass
def convert_tensors_to_lists(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_lists(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensors_to_lists(item) for item in obj)
    else:
        return obj
project_root = Path(__file__).parent
sys.path.append(str(project_root))
try:
    from pinn_dataset import AvalanchePINNDataset, create_pinn_data_loaders
    from pinn_model import AvalanchePINN
    from pinn_physics_loss import AvalanchePhysicsLoss, create_physics_loss
    from pinn_trainer import EnhancedAvalanchePINNTrainer
    from pinn_evaluator import AvalancheEvaluator
    from global_data_config import GlobalDataConfig
except ImportError as e:
    print(f"[Error] Module import failed: {e}")
    print("Please ensure all optimization modules exist and are accessible")
    sys.exit(1)
def print_banner():
    print("=" * 80)
    print("Integrated PINN Avalanche Simulation Training System")
    print("   Physics-Informed Neural Networks for Avalanche Simulation")
    print("=" * 80)
    print("Unified training script with all optimizations")
    print("Optimized weights: Total loss reduced by 95%")
    print("Fixed gradient calculation and boundary conditions")
    print("Integrated training, evaluation, and monitoring")
    print("=" * 80)
def setup_logging(log_dir: str, log_level: str = 'INFO') -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    logging.info(f"Logging system initialized")
    logging.info(f"Log file: {log_file}")
def load_config(config_path: str) -> Dict[str, Any]:
    script_dir = Path(__file__).parent
    if not os.path.isabs(config_path):
        possible_paths = [
            script_dir / config_path,  
            script_dir / "configs" / "pinn_config.yaml",  
            Path(os.getcwd()) / config_path,  
            script_dir.parent / config_path,  
        ]
        config_path_obj = None
        for path in possible_paths:
            if path.exists():
                config_path_obj = path
                break
        if config_path_obj is None:
            error_msg = f"Config file not found.Tried the following paths:\n"
            for i, path in enumerate(possible_paths, 1):
                error_msg += f"  {i}. {path}\n"
            error_msg += f"\nEnsure config file exists in any of these paths."
            raise FileNotFoundError(error_msg)
        config_path = config_path_obj
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logging.info(f"Config loaded successfully: {config_path}")
    return config
def validate_config(config: Dict[str, Any]) -> None:
    required_sections = [
        , 'model', 'training', 'physics', 'paths', 'physics_loss'
    ]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config: {section}")
    data_config = config['data']
    h5_file = Path(data_config['h5_file_path'])
    if not h5_file.exists():
        raise FileNotFoundError(f"H5 data file does not exist: {h5_file}")
    norm_file = Path(data_config['normalization_file'])
    if not norm_file.exists():
        raise FileNotFoundError(f"Normalization file does not exist: {norm_file}")
    logging.info("Configuration validated")
def create_directories(config: Dict[str, Any]) -> None:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    paths_cfg = config.get('paths', {})
    base_ckpt = Path(paths_cfg.get('checkpoints_dir', './checkpoints'))
    base_results = Path(paths_cfg.get('results_dir', './results'))
    base_logs = Path(paths_cfg.get('log_dir', './logs'))
    base_eval = Path(paths_cfg.get('evaluation_dir', './evaluation'))
    ckpt_dir = base_ckpt / ts
    results_dir = base_results / ts
    eval_dir = base_eval / ts
    config['paths']['checkpoints_dir'] = str(ckpt_dir)
    config['paths']['results_dir'] = str(results_dir)
    config['paths']['evaluation_dir'] = str(eval_dir)
    config['paths']['log_dir'] = str(base_logs)  
    for dir_path in [ckpt_dir, results_dir, base_logs, eval_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating directory: {dir_path}")
def check_gpu_availability(config: Dict[str, Any]) -> torch.device:
    prefer_auto = config.get('device', {}).get('auto_detect', True)
    device_id = int(config.get('device', {}).get('device_id', 0))
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(device_id if device_id < gpu_count else torch.cuda.current_device())
        gpu_props = torch.cuda.get_device_properties(device_id if device_id < gpu_count else torch.cuda.current_device())
        gpu_memory = gpu_props.total_memory / 1024**3
        torch.cuda.empty_cache()
        device = torch.device(f'cuda:{device_id}') if not prefer_auto else torch.device('cuda')
        logging.info(f"Device: CUDA({gpu_name}, {gpu_memory:.1f}GB), GPUs={gpu_count}, using: {device}")
    else:
        device = torch.device('cpu')
        logging.info("Device: CPU (CUDA unavailable in current environment)")
        logging.info("To use GPU, run in CUDA-enabled environment or use correct python path")
    return device
def print_training_summary(config: Dict[str, Any], device: torch.device):
    print("\n[Config] Training configuration summary")
    print("=" * 60)
    print(f"[Target] Training mode: {config['training'].get('mode', 'joint_training')}")
    print(f"[Update] Training epochs: {config['training']['num_epochs']}")
    print(f"[Batch] Batch size: {config['training']['batch_size']}")
    print(f"[Trend] Learning rate: {config['training'].get('initial_lr', 'N/A')}")
    physics_loss = config.get('physics_loss', {})
    print(f"\n[Control] Optimized weight characteristics:")
    print(f"  - initial constraint weights: {physics_loss.get('initial_constraint_weight', 'N/A')}")
    print(f"  - boundary constraint weights: {physics_loss.get('boundary_constraint_weight', 'N/A')}")
    print(f"  - max Loss value: {physics_loss.get('max_loss_value', 'N/A')}")
    model_config = config['model']
    hidden_dim = model_config.get('hidden_dim', model_config.get('hidden_width', 'N/A'))
    num_layers = model_config.get('num_hidden_layers', model_config.get('hidden_layers', 'N/A'))
    input_channels = model_config.get('input_channels', model_config.get('input_dim', 'N/A'))
    output_channels = model_config.get('output_channels', model_config.get('output_dim', 'N/A'))
    print(f"\n[Model] Model architecture: {hidden_dim}x{num_layers}")
    print(f"[Stats] Input channels: {input_channels}")
    print(f"[Stats] Output channels: {output_channels}")
    print(f"[Data] Data file: {Path(config['data']['h5_file_path']).name}")
    print(f"[Train] Training tiles: {len(config['data']['train_tile_ids'])}  ")
    print(f"[OK] Validation tiles: {len(config['data']['val_tile_ids'])}  ")
    print(f"[Compute] Computing device: {device}")
    print("=" * 60)
def print_concise_startup_info(config: Dict[str, Any], device: torch.device):
    train_cfg = config.get('training', {})
    opt_cfg = config.get('optimizer', {})
    sch_cfg = config.get('scheduler', {})
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    phys_loss_cfg = config.get('physics_loss', {})
    mode = train_cfg.get('mode', 'progressive')
    epochs = train_cfg.get('num_epochs')
    batch = train_cfg.get('batch_size')
    lr = train_cfg.get('initial_lr', opt_cfg.get('lr'))
    opt_type = opt_cfg.get('type', 'AdamW')
    sch_type = sch_cfg.get('type', 'CosineAnnealingLR')
    input_ch = model_cfg.get('input_channels', model_cfg.get('input_dim', '?'))
    output_ch = model_cfg.get('output_channels', model_cfg.get('output_dim', '?'))
    hidden_dim = model_cfg.get('hidden_dim', model_cfg.get('hidden_width', '?'))
    hidden_layers = model_cfg.get('num_hidden_layers', model_cfg.get('hidden_layers', '?'))
    dx = data_cfg.get('dx', '?')
    dy = data_cfg.get('dy', '?')
    dt = data_cfg.get('dt', '?')
    h5_name = Path(data_cfg.get('h5_file_path', 'data.h5')).name
    train_tiles = data_cfg.get('train_tile_ids', [])
    val_tiles = data_cfg.get('val_tile_ids', [])
    cont_w = phys_loss_cfg.get('continuity_weight', 1.0)
    mx_w = phys_loss_cfg.get('momentum_x_weight', 1.0)
    my_w = phys_loss_cfg.get('momentum_y_weight', 1.0)
    init_w = phys_loss_cfg.get('initial_constraint_weight', 2.0)
    bnd_w = phys_loss_cfg.get('boundary_constraint_weight', 1.0)
    max_loss = phys_loss_cfg.get('max_loss_value', 50.0)
    data_top_w = train_cfg.get('data_loss_weight', 1.0)
    phys_top_w = train_cfg.get('physics_loss_weight', 1.0)
    print(f"Starting training: Mode={mode}, epochs={epochs}, batch={batch}, lr={lr}, Opt={opt_type}, Sched={sch_type}")
    print(f"model: in={input_ch}, out={output_ch}, hidden={hidden_dim}x{hidden_layers}")
    print(f"data: {h5_name}, tiles(train={len(train_tiles)}, val={len(val_tiles)}), dx={dx}m dy={dy}m dt={dt}s")
    print(f"weights: data={data_top_w}, physics={phys_top_w}")
    print(f"Physics Loss: continuity={cont_w}, momentum_x={mx_w}, momentum_y={my_w}, initial={init_w}, boundary={bnd_w}, max_loss={max_loss}")
    print(f"Device: {device}")
def initialize_model(config: Dict[str, Any], device: torch.device) -> AvalanchePINN:
    model_config = config['model']
    input_channels = model_config.get('input_channels', model_config.get('input_dim', 14))
    output_channels = model_config.get('output_channels', model_config.get('output_dim', 3))
    hidden_dim = model_config.get('hidden_dim', model_config.get('hidden_width', 128))
    num_hidden_layers = model_config.get('num_hidden_layers', model_config.get('hidden_layers', 4))
    dropout_rate = model_config.get('dropout_rate', 0.25)
    use_dropout = model_config.get('use_dropout', dropout_rate > 0)
    use_batch_norm = model_config.get('use_batch_norm', False)
    out_constraints = model_config.get('output_constraints', {})
    max_velocity = out_constraints.get('velocity_clipping', model_config.get('max_velocity', 50.0))
    enable_output_constraints = out_constraints.get('height_nonnegative', model_config.get('enable_output_constraints', True))
    model = AvalanchePINN(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate,
        max_velocity=max_velocity,
        enable_output_constraints=enable_output_constraints,
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"model initialization completed:")
    logging.info(f"  Total Parameters: {total_params:,}")
    logging.info(f"  Trainable Parameters: {trainable_params:,}")
    logging.info(f"  Device: {device}")
    return model
def initialize_physics_loss(config: Dict[str, Any], device: torch.device) -> AvalanchePhysicsLoss:
    data_config = config.get('data', {})
    h5_file_path = data_config.get('h5_file_path')
    if not h5_file_path:
        raise ValueError("Missing h5_file_path in configuration")
    norm_file_path = data_config.get('normalization_file')
    global_data_config = GlobalDataConfig(h5_file_path=h5_file_path, normalization_file_path=norm_file_path)
    physics_loss = create_physics_loss(config, global_data_config, device)
    physics_loss.to(device)
    physics_loss_cfg = config.get('physics_loss', {})
    data_cfg = config.get('data', {})
    phys_cfg = config.get('physics', {})
    logging.info("Physics Loss module initialization completed (factory):")
    logging.info(f"  Grid spacing: dx={data_cfg.get('dx','?')}m, dy={data_cfg.get('dy','?')}m, dt={data_cfg.get('dt','?')}s")
    logging.info(
         % (
            getattr(physics_loss, 'dx', '?'),
            getattr(physics_loss, 'dy', '?'),
            getattr(physics_loss, 'dt', '?')
        )
    )
    try:
        dx_cfg = float(data_cfg['dx']) if 'dx' in data_cfg else None
        dy_cfg = float(data_cfg['dy']) if 'dy' in data_cfg else None
        dt_cfg = float(data_cfg['dt']) if 'dt' in data_cfg else None
        dx_act = float(physics_loss.dx) if hasattr(physics_loss, 'dx') else None
        dy_act = float(physics_loss.dy) if hasattr(physics_loss, 'dy') else None
        dt_act = float(physics_loss.dt) if hasattr(physics_loss, 'dt') else None
        if None not in (dx_cfg, dy_cfg, dt_cfg, dx_act, dy_act, dt_act):
            if (dx_cfg != dx_act) or (dy_cfg != dy_act) or (dt_cfg != dt_act):
                logging.warning(
                )
    except Exception:
        pass
    logging.info(f"  Enhanced constraints enabled: {physics_loss_cfg.get('enable_enhanced_constraints', True)}")
    logging.info(f"  PDE pixel sampling ratio: active={physics_loss_cfg.get('pde_active_pixel_ratio', None)}, inactive={physics_loss_cfg.get('pde_inactive_pixel_ratio', None)}")
    logging.info(f"  Active sampling: threshold={physics_loss_cfg.get('active_height_threshold', phys_cfg.get('active_sampling', {}).get('height_threshold', 0.05))}, ratio={physics_loss_cfg.get('active_sampling_ratio', phys_cfg.get('active_sampling', {}).get('active_ratio', 0.7))}")
    logging.info(f"  optimized weights: initial={physics_loss_cfg.get('initial_constraint_weight', 2.0)}, boundary={physics_loss_cfg.get('boundary_constraint_weight', 1.0)}, maxLoss={physics_loss_cfg.get('max_loss_value', 50.0)}")
    return physics_loss
def print_actual_scales(config, physics_loss):
    data_cfg = config.get('data', {})
    dx_cfg = data_cfg.get('dx', '?')
    dy_cfg = data_cfg.get('dy', '?')
    dt_cfg = data_cfg.get('dt', '?')
    dx_act = getattr(physics_loss, 'dx', '?')
    dy_act = getattr(physics_loss, 'dy', '?')
    dt_act = getattr(physics_loss, 'dt', '?')
    print(
    )
    print("Note: Trainer uses time_scale = (num_time_steps - 1) * dt for denormalization; if metadata is unavailable, time_scale = dt is used.")
def train_model(
    config: Dict[str, Any],
    model: AvalanchePINN,
    physics_loss: AvalanchePhysicsLoss,
    device: torch.device
) -> Dict[str, Any]:
    logging.info("Creating data loaders...")
    sampling_config = config['data'].get('sampling', {})
    num_data_points = config['data'].get('num_data_points', sampling_config.get('data_points_per_batch', 5000))
    num_physics_points = config['data'].get('num_physics_points', sampling_config.get('physics_points_per_batch', 3000))
    data_config = config.get('data', {})
    h5_file_path = data_config.get('h5_file_path')
    norm_file_path = data_config.get('normalization_file')
    global_data_config = GlobalDataConfig(h5_file_path=h5_file_path, normalization_file_path=norm_file_path)
    train_loader, val_loader, test_loader = create_pinn_data_loaders(
        h5_file_path=config['data']['h5_file_path'],
        normalization_stats_path=config['data']['normalization_file'],
        global_data_config=global_data_config,
        train_tiles=config['data']['train_tile_ids'],
        val_tiles=config['data']['val_tile_ids'],
        test_tiles=config['data']['test_tile_ids'],
        batch_size=config['training']['batch_size'],
        num_data_points=num_data_points,
        num_physics_points=num_physics_points,
        num_workers=config['training'].get('num_workers', 4),
        prefetch_factor=config['training'].get('prefetch_factor', 2),
        pin_memory=config['training'].get('pin_memory', True)
    )
    logging.info(f"dataloader creation completed:")
    logging.info(f"  training set: {len(train_loader)} batch")
    logging.info(f"  validation set: {len(val_loader)} batch")
    logging.info(f"  test set: {len(test_loader)} batch")
    trainer = EnhancedAvalanchePINNTrainer(
        model=model,
        physics_loss=physics_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    logging.info("starting model training...")
    training_history = trainer.train(start_epoch=0)
    logging.info("modelTraining completed")
    try:
        results_dir = Path(config['paths']['results_dir'])
        results_dir.mkdir(exist_ok=True)
        loss_curve_path = results_dir / 'training_loss_curve.png'
        trainer.plot_training_history(save_path=str(loss_curve_path))
        logging.info(f"Train Loss curve saved to: {loss_curve_path}")
    except Exception as plot_err:
        logging.warning(f"failed to plot Train Loss curve: {plot_err}")
    return training_history
def evaluate_model(
    config: Dict[str, Any],
    model: AvalanchePINN,
    physics_loss: AvalanchePhysicsLoss,
    device: torch.device
) -> Dict[str, Any]:
    logging.info("Creating test dataloader...")
    data_config = config.get('data', {})
    h5_file_path = data_config.get('h5_file_path')
    norm_file_path = data_config.get('normalization_file')
    global_data_config = GlobalDataConfig(h5_file_path=h5_file_path, normalization_file_path=norm_file_path)
    train_loader, val_loader, test_loader = create_pinn_data_loaders(
        h5_file_path=config['data']['h5_file_path'],
        normalization_stats_path=config['data']['normalization_file'],
        global_data_config=global_data_config,
        train_tiles=config['data']['train_tile_ids'],
        val_tiles=config['data']['val_tile_ids'],
        test_tiles=config['data']['test_tile_ids'],
        batch_size=config['training']['batch_size'],
        num_data_points=config['data'].get('num_data_points', 5000),
        num_physics_points=config['data'].get('num_physics_points', 3000),
        num_workers=config['training'].get('num_workers', 4)
    )
    evaluator = AvalancheEvaluator(
        model=model,
        physics_loss=physics_loss,
        device=device,
        output_dir=config['paths'].get('evaluation_dir', 'test_evaluation')
    )
    logging.info("Starting single-step error evaluation...")
    single_step_results = evaluator.single_step_evaluation(
        test_loader=test_loader,
        save_results=True
    )
    logging.info("Starting multi-step rollout predictionevaluation...")
    for batch in test_loader:
        initial_state = batch['input'][:1].to(device)
        break
    multi_step_results = evaluator.multi_step_rollout(
        initial_state=initial_state,
        num_steps=config.get('evaluation', {}).get('rollout_steps', 10),
        save_intermediate=True
    )
    try:
        logging.info("Generating visual comparison plots for validation set...")
        evaluator.visualize_from_loader(val_loader, save_prefix='val_visual', max_batches=1, max_samples=2)
        logging.info("Generating sequence maps and GIFs for validation set...")
        evaluator.visualize_val_sequence(val_loader, steps=10, samples_per_step=1, save_prefix='val_sequence')
    except Exception as viz_err:
        logging.warning(f"Validation set visualization failed: {viz_err}")
    report = evaluator.generate_evaluation_report(
        single_step_results=single_step_results,
        multi_step_results=multi_step_results,
        save_path=os.path.join(config['paths'].get('evaluation_dir', 'test_evaluation'), 'evaluation_report.txt')
    )
    return {
        : single_step_results,
        : multi_step_results,
        : report
    }
def parse_arguments():
    parser = argparse.ArgumentParser(description='PINN Avalanche Simulation Training System')
    parser.add_argument('--config', type=str, default='configs/pinn_config.yaml',
                       help='Config file path')
    parser.add_argument('--epochs', type=int, help='Training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--eval-only', action='store_true',
                       help='Execute evaluation only, no training')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level')
    parser.add_argument('--verbose-startup', action='store_true',
                       help='Detailed startup info (banner and full config summary)')
    return parser.parse_args()
def main():
    args = parse_arguments()
    if args.verbose_startup:
        print_banner()
    try:
        config = load_config(args.config)
        if args.epochs:
            config['training']['num_epochs'] = args.epochs
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if args.lr:
            config['training']['initial_lr'] = args.lr
        validate_config(config)
        create_directories(config)
        setup_logging(config['paths']['log_dir'], args.log_level)
        device = check_gpu_availability(config)
        if args.verbose_startup:
            print_training_summary(config, device)
        else:
            print_concise_startup_info(config, device)
        model = initialize_model(config, device)
        physics_loss = initialize_physics_loss(config, device)
        print_actual_scales(config, physics_loss)
        if args.resume:
            checkpoint_path = Path(args.resume)
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"Resuming model from checkpoint: {checkpoint_path}")
            else:
                logging.warning(f"Checkpoint file not found: {checkpoint_path}")
        if args.eval_only:
            logging.info("Executing model evaluation...")
            evaluation_results = evaluate_model(config, model, physics_loss, device)
            print("\n[OK] Model evaluation completed")
            print(f"[Stats] Evaluation results saved in: {config['paths'].get('evaluation_dir', 'test_evaluation')}")
        else:
            logging.info("starting model training...")
            training_history = train_model(config, model, physics_loss, device)
            results_dir = Path(config['paths']['results_dir'])
            results_dir.mkdir(exist_ok=True)
            torch.save(training_history, results_dir / 'training_history.pth')
            json_history = convert_tensors_to_lists(training_history)
            with open(results_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
                json.dump(json_history, f, indent=2, ensure_ascii=False)
            print("\n[OK] Training completed")
            print(f"[Stats] Training history saved in: {results_dir}")
            print(f"[Best] Best model saved in: {config['paths']['checkpoints_dir']}")
            try:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                simple_out = Path(config['paths'].get('project_root', '.')) / f"evaluation_simple_{ts}"
                simple_out.mkdir(exist_ok=True)
                data_cfg = config.get('data', {})
                h5_file_path = data_cfg.get('h5_file_path')
                norm_file_path = data_cfg.get('normalization_file')
                global_data_config = GlobalDataConfig(h5_file_path=h5_file_path, normalization_file_path=norm_file_path)
                augment_cfg = config.get('training', {}).get('augment', {})
                _, val_loader, _ = create_pinn_data_loaders(
                    h5_file_path=h5_file_path,
                    normalization_stats_path=norm_file_path,
                    global_data_config=global_data_config,
                    train_tiles=data_cfg.get('train_tile_ids', []),
                    val_tiles=data_cfg.get('val_tile_ids', []),
                    test_tiles=[],
                    batch_size=config['training']['batch_size'],
                    num_data_points=data_cfg.get('num_data_points', 5000),
                    num_physics_points=data_cfg.get('num_physics_points', 3000),
                    num_workers=config['training'].get('num_workers', 4),
                    augment_config=augment_cfg
                )
                evaluator = AvalancheEvaluator(model=model, physics_loss=physics_loss, device=device, output_dir=str(simple_out))
                simple_metrics = evaluator.single_step_evaluation(test_loader=val_loader, save_results=True)
                overview_path = simple_out / 'validation_overview.txt'
                with open(overview_path, 'w', encoding='utf-8') as f:
                    f.write(f"RMSE: {simple_metrics.get('total_rmse', 0.0):.6f}\n")
                    f.write(f"MAE: {simple_metrics.get('total_mae', 0.0):.6f}\n")
                    f.write(f"R2: {simple_metrics.get('total_r2', 0.0):.4f}\n")
                print(f"[Trend] Simple verification completed: {overview_path}")
                thr = float(config.get('training', {}).get('post_validation_viz_threshold', 0.9))
                if simple_metrics.get('total_r2', 0.0) >= thr:
                    try:
                        from pinn_model_validation import PINNValidator
                        v_out = Path(config['paths'].get('project_root', '.')) / f"evaluation_tuningB_{ts}"
                        v = PINNValidator(str(Path(config['paths']['checkpoints_dir']) / 'best_model.pth'), str(Path(config['paths']['project_root']) / 'configs' / 'pinn_config.yaml'), str(v_out), config['data']['val_tile_ids'][0] if config['data'].get('val_tile_ids') else 'tile_0011', quiet=True)
                        v.run()
                        print(f"[Image] Visual verification output: {v_out}")
                    except Exception as _:
                        pass
            except Exception as _:
                pass
    except Exception as e:
        logging.error(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()
