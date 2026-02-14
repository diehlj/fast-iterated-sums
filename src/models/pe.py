# positional_encodings_2d.py
import math
import torch
import torch.nn as nn

# ============================================================
# 1. 1D + 2D Sinusoidal Positional Encoding
# ============================================================

def sinusoidal_1d(L, dim):
    """Sinusoidal 1D positional encoding.
    
    Args:
        L: sequence length (int)
        dim: channel dimension (int)
    Returns:
        Tensor of shape (L, dim)
    """
    pos = torch.arange(L, dtype=torch.float).unsqueeze(1)
    i = torch.arange(dim, dtype=torch.float)
    angle = pos / torch.pow(10000, (2 * (i // 2)) / dim)
    emb = torch.zeros_like(angle)
    emb[:, 0::2] = torch.sin(angle[:, 0::2])
    emb[:, 1::2] = torch.cos(angle[:, 1::2])
    return emb  # (L, dim)

def _sinusoidal_2d_base(y_positions, x_positions, dim, freq_progression='geometric', 
                        base=10000, num_bands=None, max_freq=10.0):
    """Base function for 2D sinusoidal encoding with configurable frequency progression.
    
    Args:
        y_positions: y-coordinates, shape (H,) or (H, W) for broadcasting
        x_positions: x-coordinates, shape (W,) or (H, W) for broadcasting
        dim: channel dimension, must be even (int)
        freq_progression: 'geometric' or 'linear' (str)
        base: base for geometric progression (float, used when freq_progression='geometric')
        num_bands: number of frequency bands (int, used when freq_progression='linear')
        max_freq: maximum frequency (float, used when freq_progression='linear')
    Returns:
        Tensor of shape (H, W, dim)
    """
    assert dim % 2 == 0
    dim_half = dim // 2
    
    # Extract spatial dimensions and ensure positions are 1D
    if y_positions.dim() == 1:
        H = y_positions.shape[0]
        y_pos = y_positions  # (H,)
    else:
        H = y_positions.shape[0]
        y_pos = y_positions[:, 0] if y_positions.shape[-1] > 0 else y_positions.squeeze()
    
    if x_positions.dim() == 1:
        W = x_positions.shape[0]
        x_pos = x_positions  # (W,)
    else:
        W = x_positions.shape[-1] if x_positions.dim() == 2 else x_positions.shape[1]
        x_pos = x_positions[0, :] if x_positions.dim() == 2 else x_positions.squeeze()
    
    if freq_progression == 'geometric':
        # Geometric progression: angle = pos / (base ^ (2i / dim))
        # This matches the original sinusoidal_1d implementation
        i = torch.arange(dim_half, dtype=torch.float, device=y_pos.device)
        # Compute divisor for each channel: base ^ (2i / dim_half)
        divisor = torch.pow(base, (2 * (i // 2)) / dim_half)
        # Compute angles: pos / divisor
        y_angles = y_pos.unsqueeze(-1) / divisor.view(1, -1)  # (H, dim_half)
        x_angles = x_pos.unsqueeze(-1) / divisor.view(1, -1)  # (W, dim_half)
        # Apply sin/cos directly (no 2*pi multiplication)
        y_enc = torch.zeros(H, dim_half, device=y_pos.device)
        x_enc = torch.zeros(W, dim_half, device=x_pos.device)
        y_enc[:, 0::2] = torch.sin(y_angles[:, 0::2])
        y_enc[:, 1::2] = torch.cos(y_angles[:, 1::2])
        x_enc[:, 0::2] = torch.sin(x_angles[:, 0::2])
        x_enc[:, 1::2] = torch.cos(x_angles[:, 1::2])
    elif freq_progression == 'linear':
        # Linear progression: frequencies from 1.0 to max_freq
        if num_bands is None:
            num_bands = dim_half
        freqs = torch.linspace(1.0, max_freq, num_bands, device=y_pos.device)
        # Shape: (num_bands,)
        # If num_bands < dim_half, we'll repeat; if > dim_half, we'll truncate
        if num_bands < dim_half:
            # Repeat frequencies to match dim_half
            repeat_factor = (dim_half + num_bands - 1) // num_bands
            freqs = freqs.repeat(repeat_factor)[:dim_half]
        elif num_bands > dim_half:
            freqs = freqs[:dim_half]
        
        # Compute angles: 2*pi * coords * freqs (for linear progression)
        y_angles = 2 * math.pi * y_pos.unsqueeze(-1) * freqs.view(1, -1)  # (H, dim_half)
        x_angles = 2 * math.pi * x_pos.unsqueeze(-1) * freqs.view(1, -1)  # (W, dim_half)
        
        # Apply sin/cos with 2*pi multiplication
        y_enc = torch.zeros(H, dim_half, device=y_pos.device)
        x_enc = torch.zeros(W, dim_half, device=x_pos.device)
        y_enc[:, 0::2] = torch.sin(y_angles[:, 0::2])
        y_enc[:, 1::2] = torch.cos(y_angles[:, 1::2])
        x_enc[:, 0::2] = torch.sin(x_angles[:, 0::2])
        x_enc[:, 1::2] = torch.cos(x_angles[:, 1::2])
    else:
        raise ValueError(f"Unknown freq_progression: {freq_progression}")
    
    # Broadcast and concatenate y and x encodings
    y_broadcast = y_enc[:, None, :].expand(H, W, dim_half)  # (H, W, dim_half)
    x_broadcast = x_enc[None, :, :].expand(H, W, dim_half)  # (H, W, dim_half)
    enc = torch.cat([y_broadcast, x_broadcast], dim=-1)
    return enc  # (H, W, dim)

def sinusoidal_2d_geometric_prog(H, W, dim, base=10000):
    """2D sinusoidal encoding with geometric frequency progression.
    
    Uses discrete positions (0, 1, 2, ..., H-1) and (0, 1, 2, ..., W-1)
    with geometric frequency progression: 1 / (base ^ (2i / dim))
    
    Args:
        H: height (int)
        W: width (int)
        dim: channel dimension, must be even (int)
        base: positional base (float, default=10000)
    Returns:
        Tensor of shape (H, W, dim)
    """
    y_pos = torch.arange(H, dtype=torch.float)
    x_pos = torch.arange(W, dtype=torch.float)
    return _sinusoidal_2d_base(y_pos, x_pos, dim, freq_progression='geometric', base=base)

def sinusoidal_2d_linear_prog(H, W, dim, num_bands=None, max_freq=10.0):
    """2D sinusoidal encoding with linear frequency progression.
    
    Uses normalized coordinates [0, 1] with linear frequency progression
    from 1.0 to max_freq over num_bands.
    
    Args:
        H: height (int)
        W: width (int)
        dim: channel dimension, must be even (int)
        num_bands: number of frequency bands (int, default=dim//4)
        max_freq: maximum frequency (float, default=10.0)
    Returns:
        Tensor of shape (H, W, dim)
    """
    if num_bands is None:
        num_bands = dim // 4
    y_pos = torch.linspace(0, 1, H)
    x_pos = torch.linspace(0, 1, W)
    return _sinusoidal_2d_base(y_pos, x_pos, dim, freq_progression='linear', 
                                num_bands=num_bands, max_freq=max_freq)

# Backward compatibility aliases
sinusoidal_2d = sinusoidal_2d_geometric_prog

def fourier_2d(coords, num_bands=64, max_freq=10.0):
    """Backward compatibility wrapper for fourier_2d.
    
    Original function that takes normalized coordinates.
    Now implemented using the shared sinusoidal encoding logic.
    
    Args:
        coords: tensor of normalized coords in [0,1], shape (..., 2)
        num_bands: number of frequency bands (int)
        max_freq: maximum frequency (float)
    Returns:
        Tensor of shape (..., 4*num_bands)
    """
    # Handle original interface: coords can be any shape (..., 2)
    # Extract y and x coordinates
    y_coords = coords[..., 0]  # (...,)
    x_coords = coords[..., 1]  # (...,)
    
    # Compute frequencies
    freqs = torch.linspace(1.0, max_freq, num_bands, device=coords.device)
    
    # Scale coordinates by frequencies
    coords_scaled = coords.unsqueeze(-2) * freqs.view(1, -1, 1)  # (..., num_bands, 2)
    
    # Apply sin/cos
    enc = torch.cat([torch.sin(2 * math.pi * coords_scaled),
                     torch.cos(2 * math.pi * coords_scaled)], dim=-1)  # (..., num_bands, 4)
    
    return enc.flatten(-2)  # (..., 4*num_bands)


# ============================================================
# 4. Rotary Positional Encoding (2D version)
# ============================================================

def get_freqs_2d(H, W, dim_half, base=10000):
    """Compute 2D rotary frequencies.
    
    Args:
        H: height (int)
        W: width (int)
        dim_half: half of total channel dim (C//2), even (int)
        base: positional base (float)
    Returns:
        Tuple of tensors: (y_freqs, x_freqs) with shapes
        (H, dim_half/2) and (W, dim_half/2)
    """
    inv = 1.0 / (base ** (torch.arange(0, dim_half, 2).float() / dim_half))
    y = torch.einsum('i,j->ij', torch.arange(H).float(), inv)
    x = torch.einsum('i,j->ij', torch.arange(W).float(), inv)
    return y, x  # (H, dim_half/2), (W, dim_half/2)

def rotary_2d(x, freqs_y, freqs_x):
    """Apply 2D rotary positional encoding.
    
    Args:
        x: input tensor of shape (B, H, W, C), C even
        freqs_y: y-axis frequencies of shape (H, C/4)
        freqs_x: x-axis frequencies of shape (W, C/4)
    Returns:
        Tensor of shape (B, H, W, C)
    """
    B, H, W, C = x.shape
    dim_half = C // 2
    x = x.view(B, H, W, dim_half, 2)
    cy, sy = freqs_y.cos(), freqs_y.sin()  # (H, dim_half/2)
    cx, sx = freqs_x.cos(), freqs_x.sin()  # (W, dim_half/2)
    
    # Reshape frequencies for proper broadcasting
    cy = cy.unsqueeze(1).unsqueeze(0)  # (1, H, 1, dim_half/2)
    sy = sy.unsqueeze(1).unsqueeze(0)  # (1, H, 1, dim_half/2)
    cx = cx.unsqueeze(0).unsqueeze(0)  # (1, 1, W, dim_half/2)
    sx = sx.unsqueeze(0).unsqueeze(0)  # (1, 1, W, dim_half/2)
    
    # Apply rotary encoding to the first half of channels (y-direction)
    x_y = torch.stack([x[..., :dim_half//2, 0] * cy - x[..., :dim_half//2, 1] * sy,
                       x[..., :dim_half//2, 0] * sy + x[..., :dim_half//2, 1] * cy], dim=-1)
    
    # Apply rotary encoding to the second half of channels (x-direction)
    x_x = torch.stack([x[..., dim_half//2:, 0] * cx - x[..., dim_half//2:, 1] * sx,
                       x[..., dim_half//2:, 0] * sx + x[..., dim_half//2:, 1] * cx], dim=-1)
    
    # Concatenate y and x encodings
    result = torch.cat([x_y.flatten(-2), x_x.flatten(-2)], dim=-1)
    return result

# ============================================================
# 6. Fourier (Continuous) Positional Encoding - now implemented via sinusoidal_2d_linear_prog
# ============================================================
# (fourier_2d is defined above as backward compatibility wrapper)

# ============================================================
# 7. Unified PE Wrapper
# ============================================================

class PE(nn.Module):
    """Unified positional encoding wrapper for all PE types."""
    
    def __init__(self, pe_type: str, dim: int, H: int = None, W: int = None, **kwargs):
        """Initialize PE layer.
        
        Args:
            pe_type: 'sinusoidal_1d', 'sinusoidal_2d'/'sinusoidal_2d_geometric_prog', 
                     'sinusoidal_2d_linear_prog', 'fourier' (alias for sinusoidal_2d_linear_prog),
                     'learned_2d', 'rotary', 'relative_bias'
            dim: channel dimension
            H, W: spatial dimensions (required for 2D encodings)
            **kwargs: additional args (num_heads, num_bands, max_freq, base, etc.)
        """
        super().__init__()
        # Handle backward compatibility aliases
        if pe_type == 'sinusoidal_2d':
            pe_type = 'sinusoidal_2d_geometric_prog'
        elif pe_type == 'fourier':
            pe_type = 'sinusoidal_2d_linear_prog'
        self.pe_type = pe_type
        self.dim = dim
        self.H, self.W = H, W
        
        
        if pe_type == 'sinusoidal_1d':
            self.pe = lambda x: x + sinusoidal_1d(x.shape[1], dim).to(x.device)
        elif pe_type == 'sinusoidal_2d_geometric_prog':
            assert H and W, "H, W required for 2D encodings"
            self.base = kwargs.get('base', 10000)
            # Pre-compute encoding at initialization (C, H, W are known)
            pe_img = sinusoidal_2d_geometric_prog(H, W, dim, base=self.base)  # (H, W, dim)
            pe_chw = pe_img.permute(2, 0, 1).unsqueeze(0)  # (1, dim, H, W)
            self.register_buffer('pe_encoding', pe_chw)
            self.pe = self._sinusoidal2d_geometric_forward
        elif pe_type == 'sinusoidal_2d_linear_prog':
            assert H and W, "H, W required for 2D encodings"
            self.num_bands = kwargs.get('num_bands', dim // 4)
            self.max_freq = kwargs.get('max_freq', 10.0)
            # Pre-compute encoding at initialization (C, H, W are known)
            pe_img = sinusoidal_2d_linear_prog(H, W, dim, num_bands=self.num_bands, 
                                                max_freq=self.max_freq)  # (H, W, dim)
            # Truncate/pad to match dim if needed
            if pe_img.shape[-1] > dim:
                pe_img = pe_img[..., :dim]
            elif pe_img.shape[-1] < dim:
                pad = torch.zeros(H, W, dim - pe_img.shape[-1])
                pe_img = torch.cat([pe_img, pad], dim=-1)
            pe_chw = pe_img.permute(2, 0, 1).unsqueeze(0)  # (1, dim, H, W)
            self.register_buffer('pe_encoding', pe_chw)
            self.pe = self._sinusoidal2d_linear_forward
        elif pe_type == 'rotary':
            assert H and W, "H, W required for 2D encodings"
            self.freqs_y, self.freqs_x = get_freqs_2d(H, W, dim // 2, kwargs.get('base', 10000))
            self.pe = self._rotary_forward
        else:
            raise ValueError(f"Unknown PE type: {pe_type}")
    
    def _rotary_forward(self, x):
        """Apply rotary encoding to (B, C, H, W),
           where C is even."""
        B, C, H, W = x.shape
        assert self.H == H, f"H mismatch: {self.H} != {H}"
        assert self.W == W, f"W mismatch: {self.W} != {W}"
        assert self.dim == C, f"dim mismatch: {self.dim} != {C}"
        X = x.permute(0, 2, 3, 1) # (B, H, W, C)
        assert X.shape == (B, H, W, C), f"X shape mismatch: {X.shape} != (B, H, W, C)"
        Y = rotary_2d(X, self.freqs_y.to(x.device), self.freqs_x.to(x.device))
        return Y.permute(0, 3, 1, 2) # (B, C, H, W)
    
    def _sinusoidal2d_geometric_forward(self, x):
        """Add pre-computed sinusoidal 2D PE with geometric progression to (B, C, H, W)."""
        B, C, H, W = x.shape
        assert self.H == H, f"H mismatch: {self.H} != {H}"
        assert self.W == W, f"W mismatch: {self.W} != {W}"
        assert self.dim == C, f"dim mismatch: {self.dim} != {C}"
        # pe_encoding is (1, dim, H, W), expand to (B, dim, H, W)
        return x + self.pe_encoding.expand(B, -1, -1, -1)
    
    def _sinusoidal2d_linear_forward(self, x):
        """Add pre-computed sinusoidal 2D PE with linear progression to (B, C, H, W)."""
        B, C, H, W = x.shape
        assert self.H == H, f"H mismatch: {self.H} != {H}"
        assert self.W == W, f"W mismatch: {self.W} != {W}"
        assert self.dim == C, f"dim mismatch: {self.dim} != {C}"
        # pe_encoding is (1, dim, H, W), expand to (B, dim, H, W)
        return x + self.pe_encoding.expand(B, -1, -1, -1)
    
    def forward(self, x):
        """Apply positional encoding.
        
        Args:
            x: input tensor, shape depends on pe_type:
               - 1D: (B, L, C) -> (B, L, C)
               - 2D: (B, C, H, W) -> (B, C, H, W) or (B, H*W, C) -> (B, H*W, C)
        """
        if self.pe_type in ['sinusoidal_2d_geometric_prog', 'sinusoidal_2d_linear_prog', 'rotary']:
            # (B, C, H, W)
            assert len(x.shape) == 4, f"x shape mismatch: {x.shape} != (B, C, H, W)"
            return self.pe(x)
        elif self.pe_type in ['sinusoidal_1d']:
            # Expect (B, L, C)
            assert len(x.shape) == 3, f"x shape mismatch: {x.shape} != (B, L, C)"
            return self.pe(x)
        else:
            raise ValueError(f"Unknown PE type: {self.pe_type}")

import pytest
if __name__ == "__main__":
    dim = 8
    x = torch.randn(2, dim, 14, 14)
    pe = PE('rotary', dim=dim, H=14, W=14)
    y = pe(x)
    assert y.shape == x.shape
    with pytest.raises(RuntimeError):
        dim = 7 # odd should fail
        x = torch.randn(2, dim, 14, 14)
        pe = PE('rotary', dim=dim, H=14, W=14)
        y = pe(x)