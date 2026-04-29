# models/latent_encoder.py
"""
Latent Multi-Scale ODE Encoder with Attention
"""
from typing import Optional
import torch
from torch import nn, Tensor

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError as exc:
    raise ImportError(
        "torchdiffeq is required for LatentMultiScaleODEEncoder. "
        "Install it via: pip install torchdiffeq"
    ) from exc

from .ode_blocks import MultiScaleTimeEncoding, MultiScaleODEFunc
from .attention_modules import ObservationEncoder, MultiScaleCrossAttention


class LatentMultiScaleODEEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 n_scales: int = 3,
                 n_latent_points: int = 32,
                 time_encoder: Optional[MultiScaleTimeEncoding] = None,
                 scale_emb_dim: int = 16,
                 mlp_hidden_dim: int = 128,
                 coupling_strength: float = 0.1,
                 time_encoder_kwargs: Optional[dict] = None,
                 ode_solver: str = "dopri5",
                 ode_rtol: float = 1e-5,
                 ode_atol: float = 1e-7,
                 ode_use_adjoint: bool = False,
                 attn_n_heads: int = 4,
                 attn_n_layers: int = 2,
                 attn_dropout: float = 0.1):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.n_latent_points = n_latent_points
        self.ode_solver = ode_solver
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        self.ode_use_adjoint = ode_use_adjoint
        
        if time_encoder is None:
            time_encoder_kwargs = time_encoder_kwargs or {}
            time_encoder = MultiScaleTimeEncoding(**time_encoder_kwargs)
        self.time_encoder = time_encoder
        

        self.ode_func = MultiScaleODEFunc(
            hidden_dim=hidden_dim,
            n_scales=n_scales,
            time_encoder=self.time_encoder,
            scale_emb_dim=scale_emb_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            coupling_strength=coupling_strength
        )
        

        self.obs_encoder = ObservationEncoder(
            input_dim=2 * input_dim,  # values + mask
            hidden_dim=n_scales * hidden_dim,  
            n_heads=attn_n_heads,
            n_layers=attn_n_layers,
            dropout=attn_dropout,
            use_causal_mask=False  # Allow bidirectional, get global information
        )
        
        # ===== Component 4: Cross-Attention (new) =====
        # Inject observation information into latent ODE trajectory
        self.cross_attn = MultiScaleCrossAttention(
            latent_dim=n_scales * hidden_dim,
            obs_dim=n_scales * hidden_dim,
            n_scales=n_scales,
            dropout=attn_dropout
        )
        
        # ===== Component 5: Output projection (optional) =====
        self.output_proj = nn.Sequential(
            nn.Linear(n_scales * hidden_dim, n_scales * hidden_dim),
            nn.LayerNorm(n_scales * hidden_dim)
        )
    
    def init_state(self, batch_size: int, device: torch.device, obs_init: Optional[Tensor] = None) -> Tensor:

        if obs_init is not None:
            # Initialize from observations
            return obs_init.view(batch_size, self.n_scales, self.hidden_dim)
        return torch.zeros(batch_size, self.n_scales, self.hidden_dim, device=device)
    
    def forward(self,
                times: Tensor,
                values: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:

        device = values.device
        batch_size, T, D_x = values.shape
        
        if mask is None:
            mask = values.new_ones(values.shape)
        

        obs = torch.cat([values, mask], dim=-1)  # (B, T, 2*D_x)
        
        obs_encoded = self.obs_encoder(obs, times)  # (B, T, K*D_h)
        

        t_min = times.min()
        t_max = times.max()
        latent_times = torch.linspace(
            t_min, t_max, 
            self.n_latent_points, 
            device=device
        )
        

        obs_init = obs_encoded[:, 0, :]  
        z0 = self.init_state(batch_size, device, obs_init)  # (B, K, D_h)
        

        odeint_fn = odeint_adjoint if self.ode_use_adjoint else odeint
        

        z_trajectory = odeint_fn(
            func=lambda t, z: self.ode_func(z, t, None), 
            y0=z0,
            t=latent_times,
            method=self.ode_solver,
            rtol=self.ode_rtol,
            atol=self.ode_atol
        )  # (n_latent_points, B, K, D_h)
        

        z_trajectory = z_trajectory.permute(1, 0, 2, 3)  # (B, n_latent_points, K, D_h)
        z_flat = z_trajectory.reshape(
            batch_size, self.n_latent_points, -1
        )  # (B, n_latent_points, K*D_h)
        
        # ===== Step 3: Cross-Attention to integrate observation information =====
        # Inject observation information into latent trajectory
        z_refined = self.cross_attn(
            latent=z_flat,           # Query: latent ODE trajectory
            obs=obs_encoded,         # Key/Value: encoded observations
            latent_times=latent_times,
            obs_times=times
        )  # (B, n_latent_points, K*D_h)
        

        rep = z_refined[:, -1, :]  # (B, K*D_h)
        
        # Optional output projection
        rep = self.output_proj(rep)
        
        return rep


class LatentMultiScaleODEEncoderFast(LatentMultiScaleODEEncoder):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('n_latent_points', 16)  # 32 →16
        kwargs.setdefault('ode_solver', 'euler')  # dopri5 →euler
        kwargs.setdefault('attn_n_layers', 1)     # 2 →1
        super().__init__(*args, **kwargs)