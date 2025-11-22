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


def sinusoidal_2d(H, W, dim):
    """Deterministic 2D sinusoidal encoding.

    Args:
        H: height (int)
        W: width (int)
        dim: channel dimension, must be even (int)
    Returns:
        Tensor of shape (H, W, dim)
    """
    assert dim % 2 == 0
    dim_half = dim // 2
    y_enc = sinusoidal_1d(H, dim_half)
    x_enc = sinusoidal_1d(W, dim_half)
    # Broadcast and concatenate y and x encodings
    y_broadcast = y_enc[:, None, :].expand(H, W, dim_half)  # (H, W, dim_half)
    x_broadcast = x_enc[None, :, :].expand(H, W, dim_half)  # (H, W, dim_half)
    enc = torch.cat([y_broadcast, x_broadcast], dim=-1)
    return enc  # (H, W, dim)


# ============================================================
# 2. Learned 2D Positional Embedding
# ============================================================


class Learned2DPosEmbedding(nn.Module):
    """Trainable position embeddings for fixed H×W."""

    def __init__(self, H, W, dim):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, H * W, dim))

    def forward(self, x):
        """Add learned positional embedding.

        Args:
            x: input tensor of shape (B, H*W, dim)
        Returns:
            Tensor of shape (B, H*W, dim)
        """
        return x + self.pos_emb


# ============================================================
# 3. Convolutional Positional Encoding
# ============================================================


class ConvPosEnc(nn.Module):
    """Adds position via depthwise conv (used in CPVT/ConvNeXt)."""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

    def forward(self, x):
        """Add depthwise-conv positional signal.

        Args:
            x: input tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, C, H, W)
        """
        return x + self.conv(x)


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
    y = torch.einsum("i,j->ij", torch.arange(H).float(), inv)
    x = torch.einsum("i,j->ij", torch.arange(W).float(), inv)
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
    x_y = torch.stack(
        [
            x[..., : dim_half // 2, 0] * cy - x[..., : dim_half // 2, 1] * sy,
            x[..., : dim_half // 2, 0] * sy + x[..., : dim_half // 2, 1] * cy,
        ],
        dim=-1,
    )

    # Apply rotary encoding to the second half of channels (x-direction)
    x_x = torch.stack(
        [
            x[..., dim_half // 2 :, 0] * cx - x[..., dim_half // 2 :, 1] * sx,
            x[..., dim_half // 2 :, 0] * sx + x[..., dim_half // 2 :, 1] * cx,
        ],
        dim=-1,
    )

    # Concatenate y and x encodings
    result = torch.cat([x_y.flatten(-2), x_x.flatten(-2)], dim=-1)
    return result


# ============================================================
# 5. Relative Positional Bias (2D)
# ============================================================


class Rel2DBias(nn.Module):
    """Learned bias added to attention logits (Swin/BEiT)."""

    def __init__(self, num_heads, H, W):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_heads, (2 * H - 1) * (2 * W - 1)))
        self.H, self.W = H, W

    def forward(self):
        """Return relative bias lookup.

        Returns:
            Tensor of shape (num_heads, 2H-1, 2W-1)
        """
        return self.bias.view(-1, 2 * self.H - 1, 2 * self.W - 1)


# ============================================================
# 6. Fourier (Continuous) Positional Encoding
# ============================================================


def fourier_2d(coords, num_bands=64, max_freq=10.0):
    """Continuous Fourier features for 2D coordinates.

    Args:
        coords: tensor of normalized coords in [0,1], shape (..., 2)
        num_bands: number of frequency bands (int)
        max_freq: maximum frequency (float)
    Returns:
        Tensor of shape (..., 4*num_bands)
    """
    freqs = torch.linspace(1.0, max_freq, num_bands, device=coords.device)
    coords = coords.unsqueeze(-2) * freqs.view(1, -1, 1)
    enc = torch.cat(
        [torch.sin(2 * math.pi * coords), torch.cos(2 * math.pi * coords)], dim=-1
    )
    return enc.flatten(-2)


# ============================================================
# 7. Unified PE Wrapper
# ============================================================


class PE(nn.Module):
    """Unified positional encoding wrapper for all PE types."""

    def __init__(self, pe_type: str, dim: int, H: int = None, W: int = None, **kwargs):
        """Initialize PE layer.

        Args:
            pe_type: 'sinusoidal_1d', 'sinusoidal_2d', 'learned_2d', 'conv', 'rotary', 'relative_bias', 'fourier'
            dim: channel dimension
            H, W: spatial dimensions (required for 2D encodings)
            **kwargs: additional args (num_heads, num_bands, max_freq, etc.)
        """
        super().__init__()
        self.pe_type = pe_type
        self.dim = dim
        self.H, self.W = H, W

        if pe_type == "sinusoidal_1d":
            self.pe = lambda x: x + sinusoidal_1d(x.shape[1], dim).to(x.device)
        elif pe_type == "sinusoidal_2d":
            assert H and W, "H, W required for 2D encodings"
            self.pe = self._sinusoidal2d_forward
        elif pe_type == "learned_2d":
            assert H and W, "H, W required for 2D encodings"
            self.learned_pe = Learned2DPosEmbedding(H, W, dim)
            self.pe = self.learned_pe
        elif pe_type == "conv":
            self.conv_pe = ConvPosEnc(dim, kwargs.get("kernel_size", 3))
            self.pe = self.conv_pe
        elif pe_type == "rotary":
            assert H and W, "H, W required for 2D encodings"
            self.freqs_y, self.freqs_x = get_freqs_2d(
                H, W, dim // 2, kwargs.get("base", 10000)
            )
            self.pe = self._rotary_forward
        elif pe_type == "relative_bias":
            assert H and W, "H, W required for 2D encodings"
            self.rel_bias = Rel2DBias(kwargs.get("num_heads", 8), H, W)
            self.pe = lambda x: x  # Bias used in attention, not added to features
        elif pe_type == "fourier":
            assert H and W, "H, W required for 2D encodings"
            self.num_bands = kwargs.get("num_bands", dim // 4)
            self.max_freq = kwargs.get("max_freq", 10.0)
            self.pe = self._fourier_forward
        else:
            raise ValueError(f"Unknown PE type: {pe_type}")

    def _rotary_forward(self, x):
        """Apply rotary encoding."""
        return rotary_2d(x, self.freqs_y.to(x.device), self.freqs_x.to(x.device))

    def _fourier_forward(self, x):
        """Apply Fourier encoding."""
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=x.device),
            torch.linspace(0, 1, W, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([yy, xx], dim=-1)
        fourier_pe = fourier_2d(coords, self.num_bands, self.max_freq)
        # Truncate/pad to match input channels
        if fourier_pe.shape[-1] > C:
            fourier_pe = fourier_pe[..., :C]
        elif fourier_pe.shape[-1] < C:
            pad = torch.zeros(H, W, C - fourier_pe.shape[-1], device=x.device)
            fourier_pe = torch.cat([fourier_pe, pad], dim=-1)
        return x + fourier_pe.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

    def _sinusoidal2d_forward(self, x):
        """Add sinusoidal 2D PE to (B, C, H, W)."""
        B, C, H, W = x.shape
        pe_img = sinusoidal_2d(H, W, C).to(x.device)  # (H, W, C)
        pe_chw = pe_img.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        return x + pe_chw

    def forward(self, x):
        """Apply positional encoding.

        Args:
            x: input tensor, shape depends on pe_type:
               - 1D: (B, L, C) -> (B, L, C)
               - 2D: (B, C, H, W) -> (B, C, H, W) or (B, H*W, C) -> (B, H*W, C)
        """
        if self.pe_type in ["sinusoidal_1d", "learned_2d"]:
            # Expect (B, L, C) or (B, H*W, C)
            return self.pe(x)
        elif self.pe_type in ["sinusoidal_2d", "fourier"]:
            return self.pe(x)
        else:
            # conv, rotary: expect (B, C, H, W)
            return self.pe(x)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    H, W, dim = 14, 14, 64
    # 1) Sinusoidal
    sin2d = sinusoidal_2d(H, W, dim)
    print("Sinusoidal 2D:", sin2d.shape)

    # 2) Learned
    x = torch.randn(2, H * W, dim)
    learned = Learned2DPosEmbedding(H, W, dim)
    print("Learned2D:", learned(x).shape)

    # 3) Convolutional
    x2 = torch.randn(2, dim, H, W)
    convenc = ConvPosEnc(dim)
    print("ConvPosEnc:", convenc(x2).shape)

    # 4) Rotary
    freqs_y, freqs_x = get_freqs_2d(H, W, dim // 2)
    x3 = torch.randn(2, H, W, dim)
    rope = rotary_2d(x3, freqs_y, freqs_x)
    print("Rotary2D:", rope.shape)

    # 5) Relative bias
    relbias = Rel2DBias(num_heads=8, H=H, W=W)
    print("Rel2DBias:", relbias().shape)

    # 6) Fourier
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij"
    )
    coords = torch.stack([yy, xx], dim=-1)
    fourier = fourier_2d(coords)
    print("Fourier2D:", fourier.shape)

    # Smoke tests (shapes)
    print("\n[Smoke tests]")
    # sinusoidal_1d
    assert sinusoidal_1d(L=10, dim=6).shape == (10, 6)
    # sinusoidal_2d
    s2d = sinusoidal_2d(H, W, dim)
    assert s2d.shape == (H, W, dim)
    # learned 2d
    lpe = Learned2DPosEmbedding(H, W, dim)
    x_flat = torch.randn(2, H * W, dim)
    assert lpe(x_flat).shape == (2, H * W, dim)
    # conv
    cpe = ConvPosEnc(dim)
    x_img = torch.randn(2, dim, H, W)
    assert cpe(x_img).shape == (2, dim, H, W)
    # rotary
    fy, fx = get_freqs_2d(H, W, dim // 2)
    x_hw = torch.randn(2, H, W, dim)
    assert rotary_2d(x_hw, fy, fx).shape == (2, H, W, dim)
    # rel bias
    rb = Rel2DBias(num_heads=4, H=H, W=W)
    assert rb().shape == (4, 2 * H - 1, 2 * W - 1)
    # fourier
    four = fourier_2d(coords)
    assert four.shape[-1] == 4 * 64
    # PE wrapper
    pe_s1d = PE("sinusoidal_1d", dim=dim)
    assert pe_s1d(torch.randn(2, H * W, dim)).shape == (2, H * W, dim)
    pe_s2d = PE("sinusoidal_2d", dim=dim, H=H, W=W)
    assert pe_s2d(torch.randn(2, dim, H, W)).shape == (2, dim, H, W)
    pe_learn = PE("learned_2d", dim=dim, H=H, W=W)
    assert pe_learn(torch.randn(2, H * W, dim)).shape == (2, H * W, dim)
    pe_conv = PE("conv", dim=dim)
    assert pe_conv(torch.randn(2, dim, H, W)).shape == (2, dim, H, W)
    pe_rot = PE("rotary", dim=dim, H=H, W=W)
    assert pe_rot(torch.randn(2, H, W, dim)).shape == (2, H, W, dim)
    pe_four = PE("fourier", dim=dim, H=H, W=W, num_bands=16, max_freq=6.0)
    assert pe_four(torch.randn(2, dim, H, W)).shape == (2, dim, H, W)
    print("All smoke tests passed.")
