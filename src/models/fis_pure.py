"""
FIS-only ResNet implementation for MNIST - a ResNet-style model that uses only Fast Iterated Sums (FIS) operations
instead of traditional convolutions, designed specifically for MNIST (28x28 grayscale images).
"""

import torch
import torch.nn as nn
from typing import List, Optional

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
from src.models.fis_resnet_v1 import FISLayerParameterSharingV0


class FISBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        semiring: str,
        num_nodes: int,
        pool_mode: str,
        output_size: tuple[int, int],
        match_dim: bool,
        tree_type: str = TreeType.RANDOM,
        seed: int | None = None,
        add_identity: bool = True,
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,
    ):
        super(FISBlock, self).__init__()
        
        # First FIS layer
        self.fis1 = FISLayerParameterSharingV0(
            in_channels=in_channels,
            num_trees=out_channels,
            num_nodes=2,  # FISLayerParameterSharingV0 requires exactly 2 nodes
            semiring=semiring,
            tree_type=tree_type,
            seed=seed,
            maxplus_linear=maxplus_linear,
            use_valuation=use_valuation,
            maxplus_normalization=maxplus_normalization,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second FIS layer
        self.fis2 = FISLayerParameterSharingV0(
            in_channels=out_channels,
            num_trees=out_channels,
            num_nodes=2,  # FISLayerParameterSharingV0 requires exactly 2 nodes
            semiring=semiring,
            tree_type=tree_type,
            seed=seed, # + 1000 if seed is not None else None,
            maxplus_linear=maxplus_linear,
            use_valuation=use_valuation,
            maxplus_normalization=maxplus_normalization,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Identity connection (if needed)
        self.add_identity = add_identity
        if add_identity and in_channels != out_channels:
            # Use convolution for identity connection to match output channels
            self.identity_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.identity_bn = nn.BatchNorm2d(out_channels)
        elif add_identity and in_channels == out_channels:
            self.identity_conv = None
            self.identity_bn = None
        else:
            self.identity_conv = None
            self.identity_bn = None
    
    def forward(self, x):
        identity = x
        
        # First FIS layer
        out = self.fis1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # Second FIS layer
        out = self.fis2(out)
        out = self.bn2(out)
        
        # Add identity connection if needed
        if self.add_identity:
            if self.identity_conv is not None:
                identity = self.identity_conv(identity)
                identity = self.identity_bn(identity)
            out = out + identity
        
        out = self.relu2(out)
        
        return out

class FISResNetMNIST(nn.Module):
    """
    FIS-only ResNet implementation specifically designed for MNIST.
    Uses an initial conv3x3 layer to increase channel dimension, followed by FIS blocks.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        semiring: str = Semiring.REAL,
        num_nodes: int = 2,  # FISLayerParameterSharingV0 requires exactly 2 nodes
        tree_type: str = TreeType.RANDOM,
        seed: Optional[int] = None,
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,
    ):
        super(FISResNetMNIST, self).__init__()
        
        self.num_classes = num_classes
        self.semiring = semiring
        self.num_nodes = num_nodes
        self.tree_type = tree_type
        self.seed = seed
        self.maxplus_linear = maxplus_linear
        self.use_valuation = use_valuation
        self.maxplus_normalization = maxplus_normalization

        # Initial conv3x3 layer to increase channel dimension (like in ResNet)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # FIS blocks for MNIST
        self.layer1 = FISBlock(
            in_channels=16,
            out_channels=16,
            semiring=self.semiring,
            num_nodes=self.num_nodes,
            pool_mode="avg",
            output_size=(28, 28),  # MNIST input size
            match_dim=True,
            tree_type=self.tree_type,
            seed=self.seed,
            maxplus_linear=self.maxplus_linear,
            use_valuation=self.use_valuation,
            maxplus_normalization=self.maxplus_normalization,
            add_identity=True,
        )
        
        self.layer2 = FISBlock(
            in_channels=16,
            out_channels=32,
            semiring=self.semiring,
            num_nodes=self.num_nodes,
            pool_mode="avg",
            output_size=(28, 28),  # MNIST input size
            match_dim=True,
            tree_type=self.tree_type,
            seed=self.seed,
            maxplus_linear=self.maxplus_linear,
            use_valuation=self.use_valuation,
            maxplus_normalization=self.maxplus_normalization,
            add_identity=True,
        )
        
        self.layer3 = FISBlock(
            in_channels=32,
            out_channels=64,
            semiring=self.semiring,
            num_nodes=self.num_nodes,
            pool_mode="avg",
            output_size=(28, 28),  # MNIST input size
            match_dim=True,
            tree_type=self.tree_type,
            seed=self.seed,
            maxplus_linear=self.maxplus_linear,
            use_valuation=self.use_valuation,
            maxplus_normalization=self.maxplus_normalization,
            add_identity=True,
        )

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv3x3 layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # FIS blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global average pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def fis_resnet_mnist(
    num_classes: int = 10,
    semiring: str = Semiring.REAL,
    num_nodes: int = 2,  # FISLayerParameterSharingV0 requires exactly 2 nodes
    tree_type: str = TreeType.RANDOM,
    seed: Optional[int] = None,
    maxplus_linear: bool = False,
    use_valuation: bool = False,
    maxplus_normalization: bool = False,
) -> FISResNetMNIST:
    """
    Create a FIS ResNet model for MNIST.
    
    Args:
        num_classes: Number of output classes (default: 10 for MNIST)
        semiring: Type of semiring to use
        num_nodes: Number of nodes in the trees
        tree_type: Type of tree structure
        seed: Random seed for reproducibility
        maxplus_linear: Whether to use maxplus linear operations
        use_valuation: Whether to use valuation
        maxplus_normalization: Whether to use maxplus normalization
        
    Returns:
        FISResNetMNIST model
    """
    return FISResNetMNIST(
        num_classes=num_classes,
        semiring=semiring,
        num_nodes=num_nodes,
        tree_type=tree_type,
        seed=seed,
        maxplus_linear=maxplus_linear,
        use_valuation=use_valuation,
        maxplus_normalization=maxplus_normalization,
    )


if __name__ == "__main__":
    # Test the implementation
    model = fis_resnet_mnist(
        num_classes=10,
        semiring=Semiring.REAL,
        num_nodes=2,  # FISLayerParameterSharingV0 requires exactly 2 nodes
        tree_type=TreeType.RANDOM,
        seed=42
    )
    
    # Test with MNIST input (1 channel, 28x28)
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Forward pass successful. Output shape: {output.shape}")