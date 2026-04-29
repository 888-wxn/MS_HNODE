# models/multiscale_encoder.py
"""
Multi-scale Hierarchical Neural ODE Encoder - Simplified Version

Architecture:
1. Input projection with time encoding
2. GRU encodes time series
3. Multi-scale decomposition of final hidden state
4. ODE evolution with energy coupling
5. Classification head

Key insight: Keep it simple - GRU for sequence encoding, ODE for multi-scale dynamics.
"""
from typing import Literal, Optional
import torch
from torch import nn, Tensor

try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False

from .ode_blocks import MultiScaleTimeEncoding, MultiScaleODEFunc


class MultiScaleODERNNEncoder(nn.Module):
    """
    Simplified Multi-scale ODE-RNN Encoder.
    
    Architecture:
    1. GRU encodes full time series → final hidden state
    2. Project to multi-scale representation
    3. ODE evolution with energy coupling
    4. Output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_scales: int = 3,
        scale_emb_dim: int = 16,
        mlp_hidden_dim: int = 128,
        coupling_strength: float = 0.1,
        coupling_rank: int = 16,
        n_frequencies: int = 8,
        max_period: float = 1000.0,
        poly_order: int = 2,
        ode_solver: str = "dopri5",
        ode_atol: float = 1e-3,
        ode_rtol: float = 1e-2,
        ode_adjoint: bool = False,
        encoder_mode: Literal["parallel", "sequential"] = "parallel",
        dropout: float = 0.1,
        n_reference_points: int = 8,
        use_residual: bool = True,
        tau_values: list = None,
        learnable_tau: bool = True,
        tau_min: float = 0.01,
        tau_max: float = 100.0,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.ode_solver = ode_solver
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol
        self.ode_adjoint = ode_adjoint
        self.use_residual = use_residual
        self.dropout_rate = dropout

        flat_dim = n_scales * hidden_dim
        
        # Time encoding
        self.time_encoder = MultiScaleTimeEncoding(
            n_frequencies=n_frequencies,
            max_period=max_period,
            poly_order=poly_order
        )
        time_dim = self.time_encoder.dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # GRU encoder
        self.gru = nn.GRU(
            input_size=hidden_dim + time_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )
        
        # Project GRU output to multi-scale representation
        # Input: 2*hidden_dim (bidirectional) * 2 (last + mean pooling)
        self.to_multiscale = nn.Sequential(
            nn.Linear(hidden_dim * 4, flat_dim),
            nn.LayerNorm(flat_dim),
            nn.Dropout(dropout),
        )
        
        # Multi-scale ODE function with energy coupling
        self.ode_func = MultiScaleODEFunc(
            hidden_dim=hidden_dim,
            n_scales=n_scales,
            time_encoder=self.time_encoder,
            scale_emb_dim=scale_emb_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            coupling_strength=coupling_strength,
            coupling_rank=coupling_rank,
            tau_values=tau_values,
            learnable_tau=learnable_tau,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, flat_dim),
        )
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.6)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.6)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, times: Tensor, values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.
        
        Steps:
        1. GRU encodes time series
        2. Pool to get summary representation
        3. Project to multi-scale
        4. ODE evolution
        5. Output
        """
        B, T, D = values.shape
        device = values.device
        
        if mask is None:
            mask = torch.ones_like(values)
        
        # Prepare input: concatenate values and mask
        masked_values = values * mask
        obs_input = torch.cat([masked_values, mask], dim=-1)  # (B, T, 2D)
        h_input = self.input_proj(obs_input)  # (B, T, H)
        
        # Compute time features
        delta_t = torch.zeros_like(times)
        delta_t[:, 1:] = times[:, 1:] - times[:, :-1]
        
        t_flat = times.reshape(-1)
        dt_flat = delta_t.reshape(-1)
        time_feats = self.time_encoder(t_flat, dt_flat).view(B, T, -1)
        
        # Concatenate input with time features
        gru_input = torch.cat([h_input, time_feats], dim=-1)  # (B, T, H + time_dim)
        
        # GRU encoding
        gru_out, _ = self.gru(gru_input)  # (B, T, 2*H)
        
        # Simple pooling: last state + mean (avoid max pooling to reduce overfitting)
        last_state = gru_out[:, -1, :]  # (B, 2*H)
        
        # Masked mean pooling
        obs_mask = (mask.sum(dim=-1) > 0).float().unsqueeze(-1)  # (B, T, 1)
        masked_gru = gru_out * obs_mask
        mean_state = masked_gru.sum(dim=1) / obs_mask.sum(dim=1).clamp(min=1)  # (B, 2*H)
        
        # Combine: last + mean
        pooled = torch.cat([last_state, mean_state], dim=-1)  # (B, 4*H)
        
        # Project to multi-scale representation
        z = self.to_multiscale(pooled)  # (B, n_scales * H)
        z = z.view(B, self.n_scales, self.hidden_dim)  # (B, n_scales, H)
        
        # ODE evolution
        z_evolved = self._ode_evolve(z, device)  # (B, n_scales, H)
        
        # Flatten and output
        z_flat = z_evolved.view(B, -1)  # (B, n_scales * H)
        
        return self.output_proj(z_flat)

    def _ode_evolve(self, z: Tensor, device, integration_time: float = 1.0) -> Tensor:
        """
        ODE evolution with optional residual connection.
        
        Args:
            z: (B, n_scales, H) - multi-scale states
        
        Returns:
            (B, n_scales, H) - evolved states
        """
        B, n_scales, H = z.shape
        z_input = z  # Save for residual
        
        if not TORCHDIFFEQ_AVAILABLE:
            z_evolved = self._euler_evolve(z, integration_time)
        else:
            t_span = torch.tensor([0.0, integration_time], device=device, dtype=z.dtype)
            
            n_sc = self.n_scales
            h_dim = self.hidden_dim
            ode_func = self.ode_func
            
            def ode_wrapper(t, z_in):
                z_3d = z_in.view(B, n_sc, h_dim)
                dz = ode_func(z_3d, t.item(), 0.0)
                return dz.view(B, -1)
            
            odeint_fn = odeint_adjoint if self.ode_adjoint else odeint
            
            try:
                z_2d = z.view(B, -1).clone()
                z_traj = odeint_fn(
                    ode_wrapper, z_2d, t_span,
                    method=self.ode_solver,
                    atol=self.ode_atol,
                    rtol=self.ode_rtol,
                )
                z_evolved = z_traj[-1].view(B, n_scales, H)
            except Exception:
                z_evolved = self._euler_evolve(z, integration_time)
        
        # Residual connection
        if self.use_residual:
            z_evolved = z_evolved + z_input
        
        return z_evolved

    def _euler_evolve(self, z: Tensor, total_time: float, n_steps: int = 5) -> Tensor:
        """Euler integration fallback."""
        if total_time < 0.01:
            return z
        
        dt = total_time / n_steps
        t = 0.0
        z_curr = z.clone()
        for _ in range(n_steps):
            dz = self.ode_func(z_curr, t, 0.0)
            z_curr = z_curr + dt * dz
            t += dt
        
        return z_curr

    def get_scale_representations(self, times: Tensor, values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Get individual scale representations."""
        B, T, D = values.shape
        device = values.device
        
        if mask is None:
            mask = torch.ones_like(values)
        
        masked_values = values * mask
        obs_input = torch.cat([masked_values, mask], dim=-1)
        h_input = self.input_proj(obs_input)
        
        delta_t = torch.zeros_like(times)
        delta_t[:, 1:] = times[:, 1:] - times[:, :-1]
        
        t_flat = times.reshape(-1)
        dt_flat = delta_t.reshape(-1)
        time_feats = self.time_encoder(t_flat, dt_flat).view(B, T, -1)
        
        gru_input = torch.cat([h_input, time_feats], dim=-1)
        gru_out, _ = self.gru(gru_input)
        
        last_state = gru_out[:, -1, :]
        obs_mask = (mask.sum(dim=-1) > 0).float().unsqueeze(-1)
        masked_gru = gru_out * obs_mask
        mean_state = masked_gru.sum(dim=1) / obs_mask.sum(dim=1).clamp(min=1)
        
        pooled = torch.cat([last_state, mean_state], dim=-1)
        z = self.to_multiscale(pooled).view(B, self.n_scales, self.hidden_dim)
        z = self._ode_evolve(z, device)
        
        return z

    def get_energy(self, times: Tensor, values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute energy of the representation."""
        z = self.get_scale_representations(times, values, mask)
        return self.ode_func.get_energy(z)