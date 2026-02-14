import torch
import omegaconf
import torch.nn as nn
import torch.nn.functional as F
from src.models.sums import (
    TreeDirection,
    TreeType,
    Semiring,
    create_forest,
    sum_trees,
    sum_tree,
    all_trees,
    n_vertices
)
from src.models.fis_resnet_v1 import FISLayerParameterSharingV0, FISLayerParameterSharingV2, print_tensor_stats
from src.models.pe import PE
import math

# --- helpers ---
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # print_tensor_stats(x, f"conv({self.in_ch}, {self.out_ch})")
        # print_tensor_stats(self.conv.weight, f"conv({self.in_ch}, {self.out_ch}) weights")
        x = self.bn(x)
        # print_tensor_stats(x, "bn")
        x = self.relu(x)
        # print_tensor_stats(x, "relu")
        return x

class TinyFashionNet(nn.Module):
    """
    Topology:
      (Conv-BN-ReLU) x2 -> MaxPool(2)
      (Conv-BN-ReLU) x2 -> MaxPool(2)
      (Conv-BN-ReLU) x1
      -> GlobalAvgPool -> Linear( last_channels -> 10 )
    
    With skip connections:
      - Skip connection from stem to block1b output
      - Skip connection from block2a to block2b output
    """
    def __init__(self, ch1: int, ch2: int, ch3: int, p_drop: float = 0.0,
                num_classes: int = 10, # XXX catch kwarg for generic instantiation
                use_skip_connections: bool = False
                ):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        
        self.stem = ConvBNReLU(1, ch1)                  # 28x28 -> 28x28
        self.block1b = ConvBNReLU(ch1, ch1)             # 28x28
        self.pool1 = nn.MaxPool2d(2)                    # 28x28 -> 14x14
        self.drop1 = nn.Dropout(p_drop)

        self.block2a = ConvBNReLU(ch1, ch2)             # 14x14
        self.block2b = ConvBNReLU(ch2, ch2)             # 14x14
        self.pool2 = nn.MaxPool2d(2)                    # 14x14 -> 7x7
        self.drop2 = nn.Dropout(p_drop)

        self.block3a = ConvBNReLU(ch2, ch3)             # 7x7
        self.head = nn.Linear(ch3, 10)

        # Kaiming init for convs; BN/Linear default is fine
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        # First block with potential skip connection
        stem_out = self.stem(x)
        if self.use_skip_connections:
            x = stem_out + self.block1b(stem_out)  # Skip connection: stem -> block1b
        else:
            x = self.block1b(stem_out)
        x = self.pool1(x)
        x = self.drop1(x)

        # Second block with potential skip connection
        block2a_out = self.block2a(x)
        if self.use_skip_connections:
            x = block2a_out + self.block2b(block2a_out)  # Skip connection: block2a -> block2b
        else:
            x = self.block2b(block2a_out)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.block3a(x)
        # Global Average Pool over HxW
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

class TinyFashionNetFIS(nn.Module):
    """
    TinyFashionNet with FIS layers and optional positional encoding.
    """
    def __init__(self, ch1: int, ch2: int, ch3: int, p_drop: float = 0.0,
                num_classes: int = 10, # XXX catch kwarg for generic instantiation
                use_skip_connections: bool = False,
                maxplus_linear: bool = False,
                use_valuation: bool = False,
                maxplus_normalization: bool = False,
                semiring: str = Semiring.REAL,
                use_discounted_sums: bool = False,
                discount_init: float = 0.99,
                num_nodes: int = 2,
                fis1_seed: int = 0,
                fis2_seed: int = 3,

                sums_internal_activation: str | None = None,

                # Positional encoding parameters
                use_positional_encoding: bool = False,
                pe_type: str = "sinusoidal_2d",
                pe_kwargs: dict = None,
                ):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        self.use_positional_encoding = use_positional_encoding
        assert num_nodes in [2, 3], "num_nodes must be 2 or 3"
        
        self.stem = ConvBNReLU(1, ch1)                  # 28x28 -> 28x28
        self.block1b = ConvBNReLU(ch1, ch1)             # 28x28
        self.pool1 = nn.MaxPool2d(2)                     # 28x28 -> 14x14
        self.drop1 = nn.Dropout(p_drop)

        # Positional encoding for 14x14 resolution (before FIS1)
        if use_positional_encoding:
            pe_kwargs = pe_kwargs or {}
            self.pe_14x14 = PE(pe_type, dim=ch1, H=14, W=14, **pe_kwargs)
        else:
            self.pe_14x14 = nn.Identity()

        if num_nodes == 2:
            self.fis1 = FISLayerParameterSharingV0(
                in_channels=ch1,
                num_trees=ch1,
                num_nodes=2,
                semiring=semiring,
                seed=0, # NE
                maxplus_linear=maxplus_linear,
                use_valuation=use_valuation,
                maxplus_normalization=maxplus_normalization,
                use_discounted_sums=use_discounted_sums,
                discount_init=discount_init,
                sums_internal_activation=sums_internal_activation)
        elif num_nodes == 3:
            self.fis1 = FISLayerParameterSharingV2(
                in_channels=ch1,
                num_trees=ch1,
                num_nodes=3,
                semiring=semiring,
                seed=fis1_seed,
                maxplus_linear=maxplus_linear,
                use_valuation=use_valuation,
                maxplus_normalization=maxplus_normalization,
                use_discounted_sums=use_discounted_sums,
                discount_init=discount_init,
                sums_internal_activation=sums_internal_activation)
        self.fis1_drop = nn.Dropout(p_drop)

        self.block2a = ConvBNReLU(ch1, ch2)             # 14x14
        self.block2b = ConvBNReLU(ch2, ch2)             # 14x14
        self.pool2 = nn.MaxPool2d(2)                     # 14x14 -> 7x7
        self.drop2 = nn.Dropout(p_drop)

        # Positional encoding for 7x7 resolution (before FIS2)
        if use_positional_encoding:
            pe_kwargs = pe_kwargs or {}
            self.pe_7x7 = PE(pe_type, dim=ch2, H=7, W=7, **pe_kwargs)
        else:
            self.pe_7x7 = nn.Identity()

        if num_nodes == 2:
            self.fis2 = FISLayerParameterSharingV0(
                in_channels=ch2,
                num_trees=ch2,
                num_nodes=2,
                semiring=semiring,
                seed=3, # SW
                maxplus_linear=maxplus_linear,
                use_valuation=use_valuation,
                maxplus_normalization=maxplus_normalization,
                use_discounted_sums=use_discounted_sums,
                discount_init=discount_init,
                sums_internal_activation=sums_internal_activation)
        elif num_nodes == 3:
            self.fis2 = FISLayerParameterSharingV2(
                in_channels=ch2,
                num_trees=ch2,
                num_nodes=3,
                semiring=semiring,
                seed=fis2_seed,
                maxplus_linear=maxplus_linear,
                use_valuation=use_valuation,
                maxplus_normalization=maxplus_normalization,
                use_discounted_sums=use_discounted_sums,
                discount_init=discount_init,
                sums_internal_activation=sums_internal_activation)
        self.fis2_drop = nn.Dropout(p_drop)

        self.block3a = ConvBNReLU(ch2, ch3)             # 7x7
        self.head = nn.Linear(ch3, 10)

        # Kaiming init for convs; BN/Linear default is fine
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        # First block with potential skip connection
        stem_out = self.stem(x)
        if self.use_skip_connections:
            x = stem_out + self.block1b(stem_out)  # Skip connection: stem -> block1b
        else:
            x = self.block1b(stem_out)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Add positional encoding before FIS1 (14x14 resolution)
        x_pe = self.pe_14x14(x)
        
        if self.use_skip_connections:
            x = x + self.fis1(x_pe)  # Skip connection: fis1 -> block1b
        else:
            x = self.fis1(x_pe)
        x = self.fis1_drop(x)

        # Second block with potential skip connection
        block2a_out = self.block2a(x)
        if self.use_skip_connections:
            x = block2a_out + self.block2b(block2a_out)  # Skip connection: block2a -> block2b
        else:
            x = self.block2b(block2a_out)
        x = self.pool2(x)
        x = self.drop2(x)

        # Add positional encoding before FIS2 (7x7 resolution)
        x_pe = self.pe_7x7(x)
        
        if self.use_skip_connections:
            x = x + self.fis2(x_pe)  # Skip connection: block2a -> fis2
        else:
            x = self.fis2(x_pe)
        x = self.fis2_drop(x)

        x = self.block3a(x)
        # Global Average Pool over HxW
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)
        
class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return x

class PreserveSpatialConv(nn.Module):
    """
    Single Conv2d layer that preserves spatial dimensions (height x width).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, use_relu: bool = True, groups: int = 1):
        super().__init__()
        # Calculate padding to preserve spatial dimensions: padding = (kernel_size - 1) // 2
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False, groups=groups)
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        
        # Kaiming init
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            x = self.relu(x)
        return x

class TinyFashionNetPureFIS(nn.Module):
    def __init__(self, ch1: int, ch2: int, ch3: int, p_drop: float = 0.0,
                num_classes: int = 10, # XXX catch kwarg for generic instantiation
                use_skip_connections: bool = False,
                skip_connections_gamma: float | None = None,
                maxplus_linear: bool = False,
                use_valuation: bool = False,
                maxplus_normalization: bool = False,
                semiring: str = Semiring.REAL,
                use_discounted_sums: bool = False,
                discount_init: float = 0.99,
                num_nodes: int = 2,
                fis1_seed: int = 0,
                fis2_seed: int = 3,
                fis1_fix_alpha_1: float | None = None,
                fis2_fix_alpha_1: float | None = None,

                pool_mode: str = "max", # "max" or "avg"

                sums_internal_activation: str | None = None,

                use_positional_encoding: bool = False,
                pe_type: str = "sinusoidal_2d", # XXX unused
                pe_kwargs: dict = None, # XXX unused
                ):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        self.pool_mode = pool_mode

        if use_positional_encoding:
            pe_kwargs = pe_kwargs or {}
            self.pe = PE(pe_type, dim=ch2, H=28, W=28, **pe_kwargs)
        else:
            self.pe = nn.Identity()

        if use_skip_connections:
            if skip_connections_gamma is not None:
                self.fis1_gamma = nn.Parameter(torch.ones(1) * skip_connections_gamma)
                self.fis2_gamma = nn.Parameter(torch.ones(1) * skip_connections_gamma)
            else:
                self.fis1_gamma = torch.ones(1, device='cuda')
                self.fis2_gamma = torch.ones(1, device='cuda')
            
            # Projection layers to match dimensions for skip connections
            # fis1: ch1 -> ch2, so we need to project ch1 to ch2
            self.fis1_proj = nn.Conv2d(ch1, ch2, kernel_size=1, bias=False)
            # fis2: ch2 -> ch3, so we need to project ch2 to ch3
            self.fis2_proj = nn.Conv2d(ch2, ch3, kernel_size=1, bias=False)

        # self.use_skip_connection_gamma = use_skip_connection_gamma
        assert num_nodes in [2, 3], "num_nodes must be 2 or 3"
        
        if num_nodes == 2:  
            self.fis1 = FISLayerParameterSharingV0(
                in_channels=ch1,
                num_trees=ch2,
                num_nodes=2,
                sums_internal_activation=sums_internal_activation,
                semiring=semiring,
                seed=fis1_seed,
                maxplus_linear=maxplus_linear,
                use_valuation=use_valuation,
                maxplus_normalization=maxplus_normalization,
                use_discounted_sums=use_discounted_sums,
                discount_init=discount_init,
                fix_alpha_1=fis1_fix_alpha_1)
        elif num_nodes == 3:
            self.fis1 = FISLayerParameterSharingV2(
                in_channels=ch1,
                num_trees=ch2,
                num_nodes=3,
                sums_internal_activation=sums_internal_activation,
                semiring=semiring,
                seed=fis1_seed,
                maxplus_linear=maxplus_linear,
                use_valuation=use_valuation,
                maxplus_normalization=maxplus_normalization,
                use_discounted_sums=use_discounted_sums,
                discount_init=discount_init,
                fix_alpha_1=fis1_fix_alpha_1)
        self.ln1 = LayerNormChannels(ch2)
        self.fis1_drop = nn.Dropout(p_drop)
        
        if num_nodes == 2:
            self.fis2 = FISLayerParameterSharingV0(
                in_channels=ch2,
                num_trees=ch3,
                num_nodes=2,
                sums_internal_activation=sums_internal_activation,
                semiring=semiring,
                seed=fis2_seed,
                maxplus_linear=maxplus_linear,
                use_valuation=use_valuation,
                maxplus_normalization=maxplus_normalization,
                use_discounted_sums=use_discounted_sums,
                discount_init=discount_init,
                fix_alpha_1=fis2_fix_alpha_1)
        elif num_nodes == 3:
            self.fis2 = FISLayerParameterSharingV2(
                in_channels=ch2,
                num_trees=ch3,
                num_nodes=3,
                sums_internal_activation=sums_internal_activation,
                semiring=semiring,
                seed=fis2_seed,
                maxplus_linear=maxplus_linear,
                use_valuation=use_valuation,
                maxplus_normalization=maxplus_normalization,
                use_discounted_sums=use_discounted_sums,
                discount_init=discount_init,
                fix_alpha_1=fis2_fix_alpha_1)
        self.ln2 = LayerNormChannels(ch3)
        self.fis2_drop = nn.Dropout(p_drop)
        
        self.head = nn.Linear(ch3, 10)
        
        # Kaiming init for convs; BN/Linear default is fine
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                
    def forward(self, x):
        # First block with potential skip connection
        pre_fis1 = x
        x = self.fis1(x)
        x = self.ln1(x)
        x = self.fis1_drop(x)
        if self.use_skip_connections:
            # Project pre_fis1 from ch1 to ch2 to match fis1 output
            pre_fis1_proj = self.fis1_proj(pre_fis1)
            x = pre_fis1_proj + self.fis1_gamma * x

        pre_fis2 = x
        x = self.pe(x)
        x = self.fis2(x)
        x = self.ln2(x)
        # print_tensor_stats(x, "x after fis2")
        x = self.fis2_drop(x)
        if self.use_skip_connections:
            # Project pre_fis2 from ch2 to ch3 to match fis2 output
            pre_fis2_proj = self.fis2_proj(pre_fis2)
            x = pre_fis2_proj + self.fis2_gamma * x
        # print_tensor_stats(x, "x after fis2_drop")
        # maxpool2d:
        if self.pool_mode == "max":
            x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
        elif self.pool_mode == "avg":
            x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        else:
            raise ValueError(f"Invalid pool mode: {self.pool_mode}")
        # print_tensor_stats(x, "x after adaptive_max_pool2d")
        x = self.head(x)
        return x

class TinyFashionNetPureFISConfigurable(nn.Module):
    """
    TinyFashionNetPureFIS with configurable number of layers via channels list.
    
    Architecture pattern:
    - Optional CNN at the beginning (preserves spatial dimensions)
    - Input directly to FIS layers (or after CNN if enabled)
    - For each i from 0 to len(channels)-2:
      - FIS layer: channels[i] -> channels[i+1]
      - LayerNorm -> Dropout
    - Optional positional encoding before last FIS layer
    - Adaptive pooling -> Linear head
    """
    def __init__(self, channels: list[int], p_drop: float = 0.0,
                num_classes: int = 10,
                use_skip_connections: bool = False,
                skip_connections_gamma: float | None = None,
                maxplus_linear: bool = False,
                use_valuation: bool = False,
                maxplus_normalization: bool = False,
                semiring: str = Semiring.REAL,
                use_discounted_sums: bool = False,
                discount_init: float = 0.99,
                num_nodes: int = 2,
                fis_seeds: list[int] | None = None,
                fix_alpha_1: list[float | None] | None = None,
                pool_mode: str = "max",  # "max" or "avg"
                sums_internal_activation: str | None = None,
                use_positional_encoding: bool = False,
                pe_type: str = "sinusoidal_2d",
                pe_kwargs: dict = None,
                pe_before_layer: int | None = None,  # Which FIS layer to apply PE before (None = before last)
                use_cnn: bool = False,  # Optional CNN at the beginning
                cnn_in_channels: int = 1,  # Input channels for CNN (default 1 for grayscale)
                cnn_kernel_size: int = 3,  # Kernel size for CNN
                cnn_use_relu: bool = True,  # Whether to use ReLU activation
                ):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        self.pool_mode = pool_mode
        self.channels = channels
        assert len(channels) >= 2, "channels list must have at least 2 elements"
        assert num_nodes in [2, 3], "num_nodes must be 2 or 3"
        
        # Optional CNN at the beginning
        if use_cnn:
            # Output channels should match the first FIS layer input
            self.cnn = PreserveSpatialConv(
                in_channels=cnn_in_channels,
                out_channels=channels[0],
                kernel_size=cnn_kernel_size,
                use_relu=cnn_use_relu
            )
        else:
            self.cnn = nn.Identity()
        
        num_fis_layers = len(channels) - 1
        
        # Default seeds: [0, 3, 0, 3, ...] pattern
        if fis_seeds is None:
            fis_seeds = [0 if i % 2 == 0 else 3 for i in range(num_fis_layers)]
        assert len(fis_seeds) == num_fis_layers, "fis_seeds must have length len(channels) - 1"
        
        # Default fix_alpha_1: all None
        if fix_alpha_1 is None:
            fix_alpha_1 = [None] * num_fis_layers
        assert len(fix_alpha_1) == num_fis_layers, "fix_alpha_1 must have length len(channels) - 1"
        
        # Positional encoding
        if use_positional_encoding:
            pe_kwargs = pe_kwargs or {}
            # Default to before last FIS layer (matching original behavior)
            if pe_before_layer is None:
                pe_before_layer = num_fis_layers - 1
            # Use the channel size of the layer before the PE
            pe_dim = channels[pe_before_layer]
            self.pe = PE(pe_type, dim=pe_dim, H=28, W=28, **pe_kwargs)
            self.pe_before_layer = pe_before_layer
        else:
            self.pe = nn.Identity()
            self.pe_before_layer = -1  # Never apply
        
        # Skip connections setup
        if use_skip_connections:
            if skip_connections_gamma is not None:
                self.fis_gammas = nn.ParameterList([
                    nn.Parameter(torch.ones(1) * skip_connections_gamma)
                    for _ in range(num_fis_layers)
                ])
                self.use_gamma_buffer = False
            else:
                # Use non-parameter tensors (matching original behavior)
                # Register as buffer so it moves with the model
                self.register_buffer('fis_gammas_buffer', torch.ones(num_fis_layers, 1))
                self.use_gamma_buffer = True
            
            # Projection layers to match dimensions for skip connections
            self.fis_projs = nn.ModuleList()
            for i in range(num_fis_layers):
                # Project from channels[i] to channels[i+1]
                if channels[i] != channels[i+1]:
                    self.fis_projs.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=1, bias=False))
                else:
                    self.fis_projs.append(nn.Identity())
        else:
            self.fis_gammas = None
            self.fis_projs = None
            self.use_gamma_buffer = False
        
        # Build FIS layers dynamically
        self.fis_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.fis_drops = nn.ModuleList()
        
        for i in range(num_fis_layers):
            # FIS layer: channels[i] -> channels[i+1]
            if num_nodes == 2:
                self.fis_layers.append(FISLayerParameterSharingV0(
                    in_channels=channels[i],
                    num_trees=channels[i+1],
                    num_nodes=2,
                    sums_internal_activation=sums_internal_activation,
                    semiring=semiring,
                    seed=fis_seeds[i],
                    maxplus_linear=maxplus_linear,
                    use_valuation=use_valuation,
                    maxplus_normalization=maxplus_normalization,
                    use_discounted_sums=use_discounted_sums,
                    discount_init=discount_init,
                    fix_alpha_1=fix_alpha_1[i]))
            elif num_nodes == 3:
                self.fis_layers.append(FISLayerParameterSharingV2(
                    in_channels=channels[i],
                    num_trees=channels[i+1],
                    num_nodes=3,
                    sums_internal_activation=sums_internal_activation,
                    semiring=semiring,
                    seed=fis_seeds[i],
                    maxplus_linear=maxplus_linear,
                    use_valuation=use_valuation,
                    maxplus_normalization=maxplus_normalization,
                    use_discounted_sums=use_discounted_sums,
                    discount_init=discount_init,
                    fix_alpha_1=fix_alpha_1[i]))
            
            # LayerNorm for output channels
            self.ln_layers.append(LayerNormChannels(channels[i+1]))
            self.fis_drops.append(nn.Dropout(p_drop))
        
        # Head
        self.head = nn.Linear(channels[-1], num_classes)
        
        # Kaiming init for convs; BN/Linear default is fine
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    
    def forward(self, x):
        # Apply optional CNN at the beginning (preserves spatial dimensions)
        x = self.cnn(x)
        
        for i, (fis_layer, ln_layer, fis_drop) in enumerate(zip(self.fis_layers, self.ln_layers, self.fis_drops)):
            # Store input for skip connection
            pre_fis = x
            
            # Apply positional encoding if this is the designated layer
            if i == self.pe_before_layer:
                x = self.pe(x)
            
            # FIS layer
            x = fis_layer(x)
            x = ln_layer(x)
            x = fis_drop(x)
            
            # Skip connection
            if self.use_skip_connections:
                # Project pre_fis to match output channels
                pre_fis_proj = self.fis_projs[i](pre_fis)
                if self.use_gamma_buffer:
                    gamma = self.fis_gammas_buffer[i]
                else:
                    gamma = self.fis_gammas[i]
                x = pre_fis_proj + gamma * x
        
        # Final pooling
        if self.pool_mode == "max":
            x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
        elif self.pool_mode == "avg":
            x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        else:
            raise ValueError(f"Invalid pool mode: {self.pool_mode}")
        
        x = self.head(x)
        return x

from itertools import cycle, islice

def cyclic_sublist(seq, length):
    return list(islice(cycle(seq), length))

class TinyFashionNetPureFISUltraConfigurable(nn.Module):
    """
    Ultra-configurable variant that accepts a list specifying layer types and channel dimensions.
    
    Format: [input_ch, (layer_type, params), out_ch, (layer_type, params), out_ch, ...]
    
    Layer types:
    - ('fis', semiring): FIS layer with specified semiring ('maxplus', 'real', etc.)
    - ('cnn', cnn_type): CNN layer ('channelwise' for 1x1, 'standard' for 3x3, etc.)
    - ('pool', pool_params): Pooling layer (channels must match: in_ch == out_ch)
        - String format: 'max' or 'avg' (defaults to kernel_size=2, stride=2)
        - Dict format: {'mode': 'max', 'kernel_size': 2} (stride defaults to kernel_size)
    
    Example:
        [8, ('fis', 'maxplus'), 16, ('pool', 'max'), 16, ('cnn', 'channelwise'), 4, ('fis', 'real'), 4]
        - Input: 8 channels (from cnn_in_channels, projected if needed)
        - FIS layer (maxplus): 8 -> 16 channels
        - Pooling layer (max, 2x2): 16 -> 16 channels (spatial downsampling)
        - CNN layer (channelwise, 1x1): 16 -> 4 channels
        - FIS layer (real): 4 -> 4 channels
        - Final output: 4 channels -> num_classes
    
    Example with dict format pooling:
        [8, ('fis', 'real'), 16, ('pool', {'mode': 'avg', 'kernel_size': 3}), 16, ...]
    """
    def __init__(self, layer_config: list, p_drop: float = 0.0,
                num_classes: int = 10,
                cnn_in_channels: int = 1,  # Input channels for the first layer (default 1 for grayscale)
                use_skip_connections: bool = False,
                skip_connections_gamma: float | None = None,
                maxplus_linear: bool = False,
                use_valuation: bool = False,
                maxplus_normalization: bool = False,
                use_discounted_sums: bool = False,
                discount_init: float = 0.99,
                num_nodes: int = 2,
                fis_seeds: list[int] | None = None,
                fix_alpha_1: list[float | None] | None = None,
                pool_mode: str = "max",  # "max" or "avg"
                pool_output_size: int = 1,
                sums_internal_activation: str | None = None,
                use_positional_encoding: bool = False,
                pe_type: str = "sinusoidal_2d",
                pe_kwargs: dict = None,
                pe_before_layer: int | None = None,  # Which layer to apply PE before (None = before last)
                cnn_kernel_size: int = 3,  # Kernel size for standard CNN layers
                cnn_channelwise_kernel_size: int = 1,  # Kernel size for channelwise CNN layers (default 1 for 1x1)
                cnn_use_relu: bool = True,  # Whether to use ReLU activation in CNN layers
                use_complex: bool = False,
                complex_use_only_real_part: bool = False,
                complex_init_r_min: float = 0.0,
                complex_init_r_max: float = 1.0,
                complex_init_max_phase: float = 2 * 3.141592653589793,
                ):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        self.pool_mode = pool_mode
        self.pool_output_size = pool_output_size
        assert num_nodes in [2, 3], "num_nodes must be 2 or 3"
        
        print(f"layer_config: {layer_config} {repr(layer_config)} {type(layer_config)}")
        if isinstance(layer_config, omegaconf.listconfig.ListConfig):
            layer_config = [ list(x) if isinstance(x, omegaconf.listconfig.ListConfig) else x for x in layer_config ]
        
        # Parse layer configuration
        assert len(layer_config) >= 3, "layer_config must have at least [input_ch, (layer_type, params), out_ch]"
        assert isinstance(layer_config[0], int), "First element must be input channel dimension (int)"
        
        # Parse the configuration into a list of layer specs
        layers = []
        current_ch = layer_config[0]
        i = 1
        while i < len(layer_config):
            # Expect a tuple or list (layer_type, params) - YAML loads as lists
            layer_spec = layer_config[i]
            assert isinstance(layer_spec, (tuple, list)), f"Expected tuple or list at index {i}, got {type(layer_spec)}"
            assert len(layer_spec) == 2, f"Layer spec at index {i} must have 2 elements (layer_type, params), got {len(layer_spec)}"
            layer_type, layer_params = layer_spec
            i += 1
            
            # Expect output channels
            assert i < len(layer_config), f"Expected output channel dimension after layer spec at index {i-1}"
            assert isinstance(layer_config[i], int), f"Expected int for output channels at index {i}, got {type(layer_config[i])}"
            out_ch = layer_config[i]
            i += 1
            
            # Validate pooling layers: channels must match
            if layer_type == 'pool':
                assert current_ch == out_ch, f"Pooling layer at index {i-2} must have matching input and output channels (got {current_ch} -> {out_ch})"
            
            layers.append({
                'type': layer_type,
                'params': layer_params,
                'in_ch': current_ch,
                'out_ch': out_ch
            })
            current_ch = out_ch
        
        self.layers = layers
        num_layers = len(layers)
        
        # Count FIS layers for seeds and fix_alpha_1
        fis_layer_indices = [i for i, layer in enumerate(layers) if layer['type'] == 'fis']
        num_fis_layers = len(fis_layer_indices)
        
        # Default seeds: [0, 3, 0, 3, ...] pattern
        if fis_seeds is None:
            fis_seeds = [0 if i % 2 == 0 else 3 for i in range(num_fis_layers)]
        if len(fis_seeds) < num_fis_layers:
            print(f"[WARNING] fis_seeds has length {len(fis_seeds)} but num_fis_layers is {num_fis_layers}. Using cyclic repetition of seeds.")
            fis_seeds = cyclic_sublist(fis_seeds, num_fis_layers)
        elif len(fis_seeds) > num_fis_layers:
            print(f"[WARNING] fis_seeds has length {len(fis_seeds)} but num_fis_layers is {num_fis_layers}. Using first {num_fis_layers} seeds.")
            fis_seeds = fis_seeds[:num_fis_layers]
        
        # Default fix_alpha_1: all None
        if fix_alpha_1 is None:
            fix_alpha_1 = [None] * num_fis_layers
        assert len(fix_alpha_1) == num_fis_layers, f"fix_alpha_1 must have length {num_fis_layers} (number of FIS layers)"
        
        # Positional encoding
        if use_positional_encoding:
            pe_kwargs = pe_kwargs or {}
            # Default to before last layer (matching original behavior)
            if pe_before_layer is None:
                pe_before_layer = num_layers - 1
            # Use the channel size of the layer before the PE
            pe_dim = layers[pe_before_layer]['in_ch']
            self.pe = PE(pe_type, dim=pe_dim, H=28, W=28, **pe_kwargs)
            self.pe_before_layer = pe_before_layer
        else:
            self.pe = nn.Identity()
            self.pe_before_layer = -1  # Never apply
        
        # Skip connections setup
        if use_skip_connections:
            if skip_connections_gamma is not None:
                self.layer_gammas = nn.ParameterList([
                    nn.Parameter(torch.ones(1) * skip_connections_gamma)
                    for _ in range(num_layers)
                ])
                self.use_gamma_buffer = False
            else:
                # Use non-parameter tensors (matching original behavior)
                self.register_buffer('layer_gammas_buffer', torch.ones(num_layers, 1))
                self.use_gamma_buffer = True
            
            # Projection layers to match dimensions for skip connections
            self.layer_projs = nn.ModuleList()
            for layer in layers:
                # Project from in_ch to out_ch
                if layer['in_ch'] != layer['out_ch']:
                    self.layer_projs.append(nn.Conv2d(layer['in_ch'], layer['out_ch'], kernel_size=1, bias=False))
                else:
                    self.layer_projs.append(nn.Identity())
        else:
            self.layer_gammas = None
            self.layer_projs = None
            self.use_gamma_buffer = False
        
        # Build layers dynamically
        self.net_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.drop_layers = nn.ModuleList()
        
        fis_idx = 0  # Track which FIS layer we're on for seeds/fix_alpha_1
        
        # Input projection if needed
        if cnn_in_channels != layers[0]['in_ch']:
            self.input_proj = nn.Conv2d(cnn_in_channels, layers[0]['in_ch'], kernel_size=1, bias=False)
        else:
            self.input_proj = nn.Identity()
        
        for i, layer in enumerate(layers):
            layer_type = layer['type']
            layer_params = layer['params']
            in_ch = layer['in_ch']
            out_ch = layer['out_ch']
            
            if layer_type == 'fis':
                # Parse semiring from params
                semiring = layer_params if isinstance(layer_params, str) else str(layer_params)
                # Map string to Semiring enum value (StrEnum, so values are strings)
                semiring_lower = semiring.lower()
                if semiring_lower == 'maxplus':
                    semiring_val = Semiring.MAXPLUS
                elif semiring_lower == 'real':
                    semiring_val = Semiring.REAL
                elif semiring_lower == 'maxtimes':
                    semiring_val = Semiring.MAXTIMES
                else:
                    # Try to use as-is (might already be a Semiring enum value)
                    semiring_val = semiring
                
                # Create FIS layer
                if num_nodes == 2:
                    self.net_layers.append(FISLayerParameterSharingV0(
                        in_channels=in_ch,
                        num_trees=out_ch,
                        num_nodes=2,
                        sums_internal_activation=sums_internal_activation,
                        semiring=semiring_val,
                        seed=fis_seeds[fis_idx],
                        maxplus_linear=maxplus_linear,
                        use_valuation=use_valuation,
                        maxplus_normalization=maxplus_normalization,
                        use_discounted_sums=use_discounted_sums,
                        discount_init=discount_init,
                        fix_alpha_1=fix_alpha_1[fis_idx],
                        use_complex=use_complex,
                        complex_use_only_real_part=complex_use_only_real_part,
                        complex_init_r_min=complex_init_r_min,
                        complex_init_r_max=complex_init_r_max,
                        complex_init_max_phase=complex_init_max_phase))
                elif num_nodes == 3:
                    self.net_layers.append(FISLayerParameterSharingV2(
                        in_channels=in_ch,
                        num_trees=out_ch,
                        num_nodes=3,
                        sums_internal_activation=sums_internal_activation,
                        semiring=semiring_val,
                        seed=fis_seeds[fis_idx],
                        maxplus_linear=maxplus_linear,
                        use_valuation=use_valuation,
                        maxplus_normalization=maxplus_normalization,
                        use_discounted_sums=use_discounted_sums,
                        discount_init=discount_init,
                        fix_alpha_1=fix_alpha_1[fis_idx],
                        use_complex=use_complex,
                        complex_use_only_real_part=complex_use_only_real_part,
                        complex_init_r_min=complex_init_r_min,
                        complex_init_r_max=complex_init_r_max,
                        complex_init_max_phase=complex_init_max_phase
                        ))
                
                fis_idx += 1
                # LayerNorm for FIS layers
                self.ln_layers.append(LayerNormChannels(out_ch))
                
            elif layer_type == 'cnn':
                # Parse CNN type from params
                cnn_type = layer_params if isinstance(layer_params, str) else str(layer_params)
                
                if cnn_type.lower() == 'channelwise':
                    # Channelwise/depthwise convolution (uses groups=in_channels for true channelwise)
                    self.net_layers.append(PreserveSpatialConv(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=cnn_channelwise_kernel_size,
                        use_relu=cnn_use_relu,
                        groups=in_ch  # Depthwise convolution: each input channel convolved separately
                    ))
                elif cnn_type.lower() == 'standard':
                    # Standard 3x3 (or specified kernel size) convolution
                    self.net_layers.append(PreserveSpatialConv(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=cnn_kernel_size,
                        use_relu=cnn_use_relu
                    ))
                else:
                    raise ValueError(f"Unknown CNN type: {cnn_type}. Use 'channelwise' or 'standard'")
                
                # No LayerNorm for CNN layers (they have BN internally if needed)
                self.ln_layers.append(nn.Identity())
                
            elif layer_type == 'pool':
                # Parse pooling parameters
                # Support both string format ('max' or 'avg') and dict format
                if isinstance(layer_params, str):
                    # Simple string format: 'max' or 'avg'
                    pool_mode = layer_params.lower()
                    kernel_size = 2  # Default
                elif isinstance(layer_params, dict):
                    # Dict format: {'mode': 'max', 'kernel_size': 2}
                    pool_mode = layer_params.get('mode', 'max').lower()
                    kernel_size = layer_params.get('kernel_size', 2)
                else:
                    # Try to convert to string
                    pool_mode = str(layer_params).lower()
                    kernel_size = 2
                
                # Validate pool mode
                if pool_mode not in ['max', 'avg']:
                    raise ValueError(f"Invalid pool mode: {pool_mode}. Use 'max' or 'avg'")
                
                # Create pooling layer
                # Stride defaults to kernel_size (standard PyTorch behavior)
                if pool_mode == 'max':
                    self.net_layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size))
                else:  # pool_mode == 'avg'
                    self.net_layers.append(nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size))
                
                # No LayerNorm for pooling layers
                self.ln_layers.append(nn.Identity())
            else:
                raise ValueError(f"Unknown layer type: {layer_type}. Use 'fis', 'cnn', or 'pool'")
            
            # Dropout for all layers
            self.drop_layers.append(nn.Dropout(p_drop))
        
        # Head
        if isinstance(self.pool_output_size, (list, tuple, omegaconf.listconfig.ListConfig)):
            factor = math.prod(self.pool_output_size)
        else:
            factor = self.pool_output_size**2
        self.head = nn.Linear(layers[-1]['out_ch'] * factor, num_classes)
        
        # Kaiming init for convs; BN/Linear default is fine
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    
    def forward(self, x):
        # Apply input projection if needed
        x = self.input_proj(x)
        
        for i, (net_layer, ln_layer, drop_layer) in enumerate(zip(self.net_layers, self.ln_layers, self.drop_layers)):
            # Store input for skip connection
            pre_layer = x
            
            # Apply positional encoding if this is the designated layer
            if i == self.pe_before_layer:
                x = self.pe(x)
            
            # Apply layer
            x = net_layer(x)
            
            # Apply LayerNorm (identity for CNN layers)
            x = ln_layer(x)
            
            # Apply dropout
            x = drop_layer(x)
            
            # Skip connection (skip for pooling layers since they change spatial dimensions)
            if self.use_skip_connections and self.layers[i]['type'] != 'pool':
                # Project pre_layer to match output channels
                pre_layer_proj = self.layer_projs[i](pre_layer)
                if self.use_gamma_buffer:
                    gamma = self.layer_gammas_buffer[i]
                else:
                    gamma = self.layer_gammas[i]
                x = pre_layer_proj + gamma * x
        
        # Final pooling
        if self.pool_mode == "max":
            x = F.adaptive_max_pool2d(x, self.pool_output_size)
            x = x.flatten(start_dim=1)
        elif self.pool_mode == "avg":
            x = F.adaptive_avg_pool2d(x, self.pool_output_size)
            x = x.flatten(start_dim=1)
        else:
            raise ValueError(f"Invalid pool mode: {self.pool_mode}")
        
        x = self.head(x)
        return x

# ---- presets ----
def fashionmnist_model_A(p_drop: float = 0.0, use_skip_connections: bool = False) -> nn.Module:
    """
    ~140k params, ~22M MACs / image
    Channels: (32, 64, 128)
    """
    return TinyFashionNet(32, 64, 128, p_drop=p_drop, use_skip_connections=use_skip_connections)

def fashionmnist_model_B(p_drop: float = 0.0, use_skip_connections: bool = False) -> nn.Module:
    """
    ~218k params, ~34M MACs / image
    Channels: (40, 80, 160)
    """
    return TinyFashionNet(40, 80, 160, p_drop=p_drop, use_skip_connections=use_skip_connections)

def fashionmnist_model_C(p_drop: float = 0.0, use_skip_connections: bool = False) -> nn.Module:
    """
    ~313k params, ~49M MACs / image
    Channels: (48, 96, 192)
    """
    return TinyFashionNet(48, 96, 192, p_drop=p_drop, use_skip_connections=use_skip_connections)

# ---- FIS presets ----
def fashionmnist_model_A_fis(p_drop: float = 0.0, use_skip_connections: bool = False,
                            use_positional_encoding: bool = False, pe_type: str = "sinusoidal_2d",
                            pe_kwargs: dict = None, **fis_kwargs) -> nn.Module:
    """
    FIS variant of A with optional positional encoding
    ~140k params, ~22M MACs / image
    Channels: (32, 64, 128)
    """
    return TinyFashionNetFIS(32, 64, 128, p_drop=p_drop, use_skip_connections=use_skip_connections,
                            use_positional_encoding=use_positional_encoding, pe_type=pe_type,
                            pe_kwargs=pe_kwargs, **fis_kwargs)

def fashionmnist_model_B_fis(p_drop: float = 0.0, use_skip_connections: bool = False,
                            use_positional_encoding: bool = False, pe_type: str = "sinusoidal_2d",
                            pe_kwargs: dict = None, **fis_kwargs) -> nn.Module:
    """
    FIS variant of B with optional positional encoding
    ~218k params, ~34M MACs / image
    Channels: (40, 80, 160)
    """
    return TinyFashionNetFIS(40, 80, 160, p_drop=p_drop, use_skip_connections=use_skip_connections,
                            use_positional_encoding=use_positional_encoding, pe_type=pe_type,
                            pe_kwargs=pe_kwargs, **fis_kwargs)

def fashionmnist_model_C_fis(p_drop: float = 0.0, use_skip_connections: bool = False,
                            use_positional_encoding: bool = False, pe_type: str = "sinusoidal_2d",
                            pe_kwargs: dict = None, **fis_kwargs) -> nn.Module:
    """
    FIS variant of C with optional positional encoding
    ~313k params, ~49M MACs / image
    Channels: (48, 96, 192)
    """
    return TinyFashionNetFIS(48, 96, 192, p_drop=p_drop, use_skip_connections=use_skip_connections,
                            use_positional_encoding=use_positional_encoding, pe_type=pe_type,
                            pe_kwargs=pe_kwargs, **fis_kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Depthwise-Separable conv block: (DW 3x3 -> BN -> ReLU -> PW 1x1 -> BN -> ReLU) ---
class DSConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # Kaiming init for both convs
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x

class TinyFashionNetDS(nn.Module):
    """
    Topology:
      (DSConv x2) -> MaxPool(2)
      (DSConv x2) -> MaxPool(2)
      (DSConv x1)
      -> GAP -> Linear(last_channels -> 10)
    """
    def __init__(self, ch1: int, ch2: int, ch3: int, p_drop: float = 0.0,
                num_classes: int = 10 # XXX catch kwarg for generic instantiation
                ):
        super().__init__()
        self.block1a = DSConvBNReLU(1, ch1)                 # 28x28
        self.block1b = DSConvBNReLU(ch1, ch1)               # 28x28
        self.pool1 = nn.MaxPool2d(2)                         # -> 14x14
        self.drop1 = nn.Dropout(p_drop)

        self.block2a = DSConvBNReLU(ch1, ch2)               # 14x14
        self.block2b = DSConvBNReLU(ch2, ch2)               # 14x14
        self.pool2 = nn.MaxPool2d(2)                         # -> 7x7
        self.drop2 = nn.Dropout(p_drop)

        self.block3a = DSConvBNReLU(ch2, ch3)               # 7x7
        self.head = nn.Linear(ch3, 10)

    def forward(self, x):
        x = self.block1a(x)
        x = self.block1b(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.block2a(x)
        x = self.block2b(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.block3a(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

# ---- presets (same widths as standard A/B/C; DS reduces params/MACs ≈3–4×) ----
def fashionmnist_model_A_ds(p_drop: float = 0.0) -> nn.Module:
    """
    DS variant of A
    Channels: (32, 64, 128)
    ~40–50k params, ~6–8M MACs / image (approx)
    """
    return TinyFashionNetDS(32, 64, 128, p_drop=p_drop)

def fashionmnist_model_B_ds(p_drop: float = 0.0) -> nn.Module:
    """
    DS variant of B
    Channels: (40, 80, 160)
    ~65–75k params, ~9–12M MACs / image (approx)
    """
    return TinyFashionNetDS(40, 80, 160, p_drop=p_drop)

def fashionmnist_model_C_ds(p_drop: float = 0.0) -> nn.Module:
    """
    DS variant of C
    Channels: (48, 96, 192)
    ~90–110k params, ~13–17M MACs / image (approx)
    """
    return TinyFashionNetDS(48, 96, 192, p_drop=p_drop)


# --- optional: quick sanity test ---
if __name__ == "__main__":
    x = torch.randn(2, 1, 28, 28)
    for name, ctor in [("A", fashionmnist_model_A), ("B", fashionmnist_model_B), ("C", fashionmnist_model_C)]:
        m = ctor()
        y = m(x)
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"Model {name}: out={tuple(y.shape)}, params={n_params/1e3:.1f}k")

    x = torch.randn(2, 1, 28, 28)
    for name, ctor in [("A_ds", fashionmnist_model_A_ds),
                       ("B_ds", fashionmnist_model_B_ds),
                       ("C_ds", fashionmnist_model_C_ds)]:
        m = ctor()
        y = m(x)
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{name}: out={tuple(y.shape)}, params={n_params/1e3:.1f}k")
    
    # Test skip connections
    print("\nTesting skip connections:")
    x = torch.randn(2, 1, 28, 28)
    for name, ctor in [("A", fashionmnist_model_A), ("B", fashionmnist_model_B), ("C", fashionmnist_model_C)]:
        # Test without skip connections
        m_no_skip = ctor(use_skip_connections=False)
        y_no_skip = m_no_skip(x)
        
        # Test with skip connections
        m_skip = ctor(use_skip_connections=True)
        y_skip = m_skip(x)
        
        n_params = sum(p.numel() for p in m_skip.parameters() if p.requires_grad)
        print(f"Model {name} (skip): out={tuple(y_skip.shape)}, params={n_params/1e3:.1f}k")
        print(f"  Skip connections work: {torch.allclose(y_no_skip, y_skip, atol=1e-6) == False}")  # Should be different
    
    # Test FIS models with PE
    print("\nTesting FIS models with PE:")
    x = torch.randn(2, 1, 28, 28)
    for name, ctor in [("A_fis", fashionmnist_model_A_fis), ("B_fis", fashionmnist_model_B_fis), ("C_fis", fashionmnist_model_C_fis)]:
        # Test without PE
        m_no_pe = ctor(use_positional_encoding=False)
        y_no_pe = m_no_pe(x)
        
        # Test with sinusoidal PE
        m_pe = ctor(use_positional_encoding=True, pe_type="sinusoidal_2d")
        y_pe = m_pe(x)
        
        n_params = sum(p.numel() for p in m_pe.parameters() if p.requires_grad)
        print(f"Model {name}: out={tuple(y_pe.shape)}, params={n_params/1e3:.1f}k")
        print(f"  PE works: {torch.allclose(y_no_pe, y_pe, atol=1e-6) == False}")  # Should be different