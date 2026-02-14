import torch
import torch.nn as nn


def test_conv2d():
    '''nn.Conv2d by default convolves over all channels.
       i.e. in channel direction the kernel has size equal to the number of input channels,
       the parameter kernel_size only refers to the spatial dimensions.'''
    in_planes = 3
    out_planes = 16
    kernel_size = 5
    conv = nn.Conv2d(in_planes, out_planes, kernel_size,
                     stride=1, padding=0)
    x = torch.randn(1, in_planes, 32, 32)
    y = conv(x)
    assert y.shape == (1, out_planes, 32-kernel_size+1, 32-kernel_size+1)
    assert conv.get_parameter('weight').shape == (out_planes, in_planes, kernel_size, kernel_size)
    assert conv.get_parameter('bias').shape == (out_planes,)
    return conv

    # depthwise conv shape  kernel_size x kernel_size
    # pointwise conv shape  1 x 1 x in_planes x out_planes


if __name__ == "__main__":
    test_conv2d()