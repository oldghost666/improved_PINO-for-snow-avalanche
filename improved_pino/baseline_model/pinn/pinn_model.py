
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Tuple, Optional
class EnhancedMLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        use_residual: bool = True,
        use_batch_norm: bool = False,
        use_dropout: bool = False,
    ):
        super(EnhancedMLPBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.batch_norm = nn.BatchNorm1d(output_dim) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else None
        self.use_residual = use_residual and (input_dim == output_dim)
        if use_residual and input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        else:
            self.projection = None
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x):
        identity = x
        out = self.linear(x)
        if self.use_batch_norm and self.batch_norm is not None:
            out = self.batch_norm(out)
        out = F.gelu(out)  
        if self.use_dropout and self.dropout is not None:
            out = self.dropout(out)
        if self.use_residual:
            if self.projection is not None:
                identity = self.projection(identity)
            out = out + identity
        return out
class AvalanchePINN(nn.Module):
    def __init__(
        self,
        input_channels: int = 14,
        output_channels: int = 3,
        hidden_dim: int = 128,
        num_hidden_layers: int = 8,  
        dropout_rate: float = 0.1,   
        max_velocity: float = 5.0,  
        enable_output_constraints: bool = False,
        use_residual_connections: bool = True,  
        use_batch_norm: bool = False,  
        use_dropout: bool = False      
    ):
        super(AvalanchePINN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.max_velocity = max_velocity
        self.enable_output_constraints = enable_output_constraints
        self.use_residual_connections = use_residual_connections
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        layers = []
        layers.append(EnhancedMLPBlock(
            input_channels, 
            hidden_dim, 
            dropout_rate,
            use_residual=False,  
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
        ))
        for i in range(num_hidden_layers - 1):
            layers.append(EnhancedMLPBlock(
                hidden_dim, 
                hidden_dim, 
                dropout_rate,
                use_residual=use_residual_connections,
                use_batch_norm=use_batch_norm,
                use_dropout=use_dropout,
            ))
        self.output_layer = nn.Linear(hidden_dim, output_channels)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        self.hidden_layers = nn.ModuleList(layers)
        self._log_model_info()
    def _log_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Enhanced AvalanchePINN model constructed:")
        logging.info(f"  Input channels: {self.input_channels}")
        logging.info(f"  Output channels: {self.output_channels}")
        logging.info(f"  Hidden Dimension: {self.hidden_dim}")
        logging.info(f"  Number of Hidden Layers: {self.num_hidden_layers}")
        logging.info(f"  Dropout rate: {self.dropout_rate}")
        logging.info(f"  Residual connections: {self.use_residual_connections}")
        logging.info(f"  Batch normalization: {self.use_batch_norm}")
        logging.info(f"  Dropout enabled: {self.use_dropout}")
        logging.info(f"  Max velocity clipping: {self.max_velocity} m/s")
        logging.info(f"  Total Parameters: {total_params:,}")
        logging.info(f"  Trainable Parameters: {trainable_params:,}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, channels)
        for layer in self.hidden_layers:
            x_flat = layer(x_flat)
        output_flat = self.output_layer(x_flat)
        if self.enable_output_constraints:
            output_flat = self._apply_output_constraints(output_flat)
        output = output_flat.reshape(batch_size, height, width, self.output_channels)
        output = output.permute(0, 3, 1, 2)  
        return output
    def forward_points(self, x_points: torch.Tensor) -> torch.Tensor:
        x = x_points
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        if self.enable_output_constraints:
            output = self._apply_output_constraints(output)
        return output
    def _apply_output_constraints(self, output: torch.Tensor) -> torch.Tensor:
        return output
    def get_model_info(self) -> Dict[str, any]:
        return {
            : self.input_channels,
            : self.output_channels,
            : self.hidden_dim,
            : self.num_hidden_layers,
            : self.dropout_rate,
            : self.max_velocity,
            : self.use_residual_connections,
            : self.use_batch_norm,
            : sum(p.numel() for p in self.parameters()),
            : sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
class AvalanchePINNWithPhysics(nn.Module):
    def __init__(self, base_model: AvalanchePINN):
        super(AvalanchePINNWithPhysics, self).__init__()
        self.base_model = base_model
    def forward_with_gradients(
        self, 
        x: torch.Tensor, 
        compute_spatial_gradients: bool = True,
        compute_time_gradients: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        output = self.base_model(x)
        gradients = {}
        if compute_spatial_gradients:
            batch_size, channels, height, width = output.shape
            dx = dy = 1.0
            for i, field_name in enumerate(['height', 'velocity_x', 'velocity_y']):
                field = output[:, i:i+1, :, :]  
                grad_x = self._compute_spatial_gradient(field, dim=3, spacing=dx)
                gradients[f'd{field_name}_dx'] = grad_x
                grad_y = self._compute_spatial_gradient(field, dim=2, spacing=dy)
                gradients[f'd{field_name}_dy'] = grad_y
        if compute_time_gradients:
            gradients['time_gradients_placeholder'] = True
        return output, gradients
    def _compute_spatial_gradient(
        self, 
        field: torch.Tensor, 
        dim: int, 
        spacing: float
    ) -> torch.Tensor:
        if dim == 3:  
            grad_left = (field[:, :, :, 1:2] - field[:, :, :, 0:1]) / spacing
            grad_center = (field[:, :, :, 2:] - field[:, :, :, :-2]) / (2 * spacing)
            grad_right = (field[:, :, :, -1:] - field[:, :, :, -2:-1]) / spacing
            gradient = torch.cat([grad_left, grad_center, grad_right], dim=3)
        elif dim == 2:  
            grad_top = (field[:, :, 1:2, :] - field[:, :, 0:1, :]) / spacing
            grad_center = (field[:, :, 2:, :] - field[:, :, :-2, :]) / (2 * spacing)
            grad_bottom = (field[:, :, -1:, :] - field[:, :, -2:-1, :]) / spacing
            gradient = torch.cat([grad_top, grad_center, grad_bottom], dim=2)
        else:
            raise ValueError(f"Unsupported gradient calculation dimension: {dim}")
        return gradient
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
def create_avalanche_pinn_model(config: Dict) -> AvalanchePINN:
    model_config = config.get('model', {})
    return AvalanchePINN(
        input_channels=model_config.get('input_channels', 14),
        output_channels=model_config.get('output_channels', 3),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_hidden_layers=model_config.get('num_hidden_layers', 8),  
        dropout_rate=model_config.get('dropout_rate', 0.1),
        max_velocity=model_config.get('max_velocity', 50.0),
        enable_output_constraints=model_config.get('enable_output_constraints', False),
        use_residual_connections=model_config.get('use_residual_connections', True),
        use_batch_norm=model_config.get('use_batch_norm', False),
        use_dropout=model_config.get('use_dropout', False),
    )
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = AvalanchePINN(num_hidden_layers=8, use_residual_connections=True, use_batch_norm=False, use_dropout=False, enable_output_constraints=False)
    batch_size, height, width = 2, 64, 64
    test_input = torch.randn(batch_size, 14, height, width)
    with torch.no_grad():
        output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range - Height: [{output[:, 0].min():.3f}, {output[:, 0].max():.3f}]")
    print(f"Output range - Velocity X: [{output[:, 1].min():.3f}, {output[:, 1].max():.3f}]")
    print(f"Output range - Velocity Y: [{output[:, 2].min():.3f}, {output[:, 2].max():.3f}]")
    model_info = model.get_model_info()
    print(f"Model parameters count: {model_info['total_parameters']:,}")
    print(f"Residual connections: {model_info['use_residual_connections']}")
    print(f"Batch normalization: {model_info['use_batch_norm']}")
    print(f"\nOutput constraint test:")
    print(f"Height non-negativity: all heights >= 0? {(output[:, 0] >= 0).all()}")
    print(f"Velocity clipping: max velocity <= {model.max_velocity}? {(torch.abs(output[:, 1:]) <= model.max_velocity).all()}")
