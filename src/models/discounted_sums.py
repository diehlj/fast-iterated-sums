from src.kernels import scans_discounted_cummax, scans_discounted_cumsum
from src.kernels.scans_discounted_cumsum import DiscountedCumsumN_BCHW_Complex
from src.models.sums import Semiring, TreeDirection, linear_wh, linear_wh_channelwise
import torch
from typing import Generator
import sympy
from src.models.sums import n_vertices
import os
import datetime
import pickle

type Tree = list[float | int | torch.Tensor | list[tuple[str, list]]]

def is_complex(discount: torch.Tensor) -> bool:
    """Check if discount tensor is complex (has shape (..., 2) on last dimension)."""
    return len(discount.shape) == 5 and discount.shape[-1] == 2

def to_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Convert real tensor (..., H, W) to complex tensor (..., H, W, 2)."""
    if is_complex(tensor):
        return tensor
    else:
        return torch.stack([tensor, torch.zeros_like(tensor)], dim=-1)

def complex_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two tensors, handling both real and complex cases.
    
    Args:
        a: Real (..., H, W) or complex (..., H, W, 2) tensor
        b: Real (..., H, W) or complex (..., H, W, 2) tensor
    
    Returns:
        Complex tensor (..., H, W, 2)
    """
    if not is_complex(a) and not is_complex(b):
        return a * b

    a = to_complex(a)
    b = to_complex(b)

    # Extract real and imaginary parts
    a_real = a[..., 0]
    a_imag = a[..., 1]
    b_real = b[..., 0]
    b_imag = b[..., 1]
    
    # Complex multiplication: (a_r + i*a_i) * (b_r + i*b_i) = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)
    result_real = a_real * b_real - a_imag * b_imag
    result_imag = a_real * b_imag + a_imag * b_real
    
    return torch.stack([result_real, result_imag], dim=-1)

def safe_transpose(tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Transpose tensor, handling complex tensors correctly.
    
    For complex tensors (..., H, W, 2), if dim0=-2 and dim1=-1, we actually want
    to transpose H and W (dimensions -3 and -2), not W and the complex dimension.
    """
    
    if is_complex(tensor):
        # Complex tensor: shape is (..., H, W, 2)
        # If trying to transpose -2 and -1, transpose -3 and -2 instead (H and W)
        if dim0 == -2 and dim1 == -1:
            return torch.transpose(tensor, -3, -2)
        else:
            return torch.transpose(tensor, dim0, dim1)
    else:
        # Real tensor: normal transpose
        return torch.transpose(tensor, dim0, dim1)

def safe_flip(tensor: torch.Tensor, dims: list[int]) -> torch.Tensor:
    """Flip tensor, handling complex tensors correctly.
    
    For complex tensors (..., H, W, 2), dims=[-1] should flip H (dim -3),
    and dims=[-2] should flip W (dim -2).
    """
    if is_complex(tensor):
        # Complex tensor: shape is (..., H, W, 2)
        # Map dims: -1 -> -3 (H), -2 -> -2 (W)
        mapped_dims = []
        for d in dims:
            if d == -1:
                mapped_dims.append(-3)  # Flip H instead of complex dimension
            elif d == -2:
                mapped_dims.append(-2)  # W stays the same
            else:
                mapped_dims.append(d)
        return torch.flip(tensor, dims=mapped_dims)
    else:
        # Real tensor: normal flip
        return torch.flip(tensor, dims=dims)

def ds_cummax_N(X, discount):
    return scans_discounted_cummax.DiscountedCummaxN_BCHW.apply(discount, X)

def ds_cummax_S(X, discount):
    return torch.flip(ds_cummax_N(torch.flip(X, dims=[-2]), torch.flip(discount, dims=[-2])), dims=[-2])

def ds_cummax_W(X, discount):
    return torch.transpose(ds_cummax_N(torch.transpose(X, -2, -1), torch.transpose(discount, -2, -1)), -2, -1)

def ds_cummax_E(X, discount):
    return torch.flip(ds_cummax_W(torch.flip(X, dims=[-1]), torch.flip(discount, dims=[-1])), dims=[-1])

# XXX Careful: this is only well-defined
# if the discount is constant.
def ds_cummax_NW(X, discount):
    tmp = ds_cummax_N(X, discount)
    return ds_cummax_W(tmp, discount)

def ds_cummax_SW(X, discount):
    return torch.flip(ds_cummax_NW(torch.flip(X, dims=[-2]), torch.flip(discount, dims=[-2])), dims=[-2])

def ds_cummax_NE(X, discount):
    return torch.flip(ds_cummax_NW(torch.flip(X, dims=[-1]), torch.flip(discount, dims=[-1])), dims=[-1])

def ds_cummax_SE(X, discount):
    return torch.flip(ds_cummax_NE(torch.flip(X, dims=[-2,-1]), torch.flip(discount, dims=[-2,-1])), dims=[-2,-1])


def ds_cumsum_N(X, discount):
    if is_complex(discount):
        return DiscountedCumsumN_BCHW_Complex.apply(discount, X)
    else:
        return scans_discounted_cumsum.DiscountedCumsumN_BCHW.apply(discount, X)

def ds_cumsum_S(X, discount):
    return safe_flip(ds_cumsum_N(safe_flip(X, dims=[-2]), safe_flip(discount, dims=[-2])), dims=[-2])

def ds_cumsum_W(X, discount):
    return safe_transpose(ds_cumsum_N(safe_transpose(X, -2, -1), safe_transpose(discount, -2, -1)), -2, -1)

def ds_cumsum_E(X, discount):
    return safe_flip(ds_cumsum_W(safe_flip(X, dims=[-1]), safe_flip(discount, dims=[-1])), dims=[-1])

# XXX Careful: this is only well-defined
# if the discount is constant.
def ds_cumsum_NW(X, discount):
    tmp = ds_cumsum_N(X, discount)
    return ds_cumsum_W(tmp, discount)

def ds_cumsum_SW(X, discount):
    return safe_flip(ds_cumsum_NW(safe_flip(X, dims=[-2]), safe_flip(discount, dims=[-2])), dims=[-2])

def ds_cumsum_NE(X, discount):
    return safe_flip(ds_cumsum_NW(safe_flip(X, dims=[-1]), safe_flip(discount, dims=[-1])), dims=[-1])

def ds_cumsum_SE(X, discount):
    return safe_flip(ds_cumsum_NE(safe_flip(X, dims=[-2,-1]), safe_flip(discount, dims=[-2,-1])), dims=[-2,-1])


def ds_cumsum_XX(x: torch.Tensor, XX: str, discount: torch.Tensor, semiring: str = Semiring.REAL) -> torch.Tensor:
    """Discounted cumulative sum in direction XX."""
    if semiring == Semiring.REAL:
        if XX == TreeDirection.NW:
            return ds_cumsum_NW(x, discount)
        elif XX == TreeDirection.SW:
            return ds_cumsum_SW(x, discount)
        elif XX == TreeDirection.NE:
            return ds_cumsum_NE(x, discount)
        elif XX == TreeDirection.SE:
            return ds_cumsum_SE(x, discount)
        elif XX == TreeDirection.N:
            return ds_cumsum_N(x, discount)
        elif XX == TreeDirection.S:
            return ds_cumsum_S(x, discount)
        elif XX == TreeDirection.W:
            return ds_cumsum_W(x, discount)
        elif XX == TreeDirection.E:
            return ds_cumsum_E(x, discount)
    elif semiring in [Semiring.MAXPLUS, Semiring.MAXTIMES]:
        if XX == TreeDirection.NW:
            return ds_cummax_NW(x, discount)
        elif XX == TreeDirection.SW:
            return ds_cummax_SW(x, discount)
        elif XX == TreeDirection.NE:
            return ds_cummax_NE(x, discount)
        elif XX == TreeDirection.SE:
            return ds_cummax_SE(x, discount)
        elif XX == TreeDirection.N:
            return ds_cummax_N(x, discount)
        elif XX == TreeDirection.S:
            return ds_cummax_S(x, discount)
        elif XX == TreeDirection.W:
            return ds_cummax_W(x, discount)
        elif XX == TreeDirection.E:
            return ds_cummax_E(x, discount)
    else:
        raise ValueError(
            f"Invalid semiring: {semiring}. Value must be one of {', '.join([sr for sr in Semiring])}"
        )

def sum_tree(
    x: torch.Tensor, tree: Tree, discount: torch.Tensor, semiring: str = Semiring.REAL, 
    maxplus_linear: bool = False, channel_wise: bool = False,
) -> torch.Tensor:
    """
    Calculate the discounted sum corresponding to a corner tree.

    Args:
        x: (..., channels, height, width) or (..., channels, height, width, 2) for complex
        tree: [value, [(direction, subtree), ...]]
                'subtree' is of same form, or [value, []] for a leaf.
                'direction' is a TreeDirection (one of N, NE, E, SE, S, SW, W, NW).
                'value' can be a scalar, a tensor of shape (channels,), or a function that takes x as input
                        (..., channels, height, width) and returns (..., height, width).
        discount: (..., channels, height, width) or (..., channels, height, width, 2) for complex discount tensor
        semiring: Semiring to use for the sum.  See Semiring for more details.
        maxplus_linear: If True, use tropical linear map for the node values.

    Returns:
        (..., height, width) or (..., height, width, 2) for complex
    """
    # Detect if discount is complex
    discount_is_complex = is_complex(discount)
    x_is_complex = is_complex(x)
    
    f = tree[0]
    if callable(f):
        f_x = f(x)
    elif isinstance(f, (int, float)) or (torch.is_tensor(f) and f.numel() == 1):
        f_x = f * x
    elif torch.is_tensor(f):
        if channel_wise:
            f_x = linear_wh_channelwise(x, f, maxplus_linear=maxplus_linear, is_complex=x_is_complex)
        else:
            f_x = linear_wh(x, f, maxplus_linear=maxplus_linear, is_complex=x_is_complex)
            # XXX argggh .. we need to do somehting for complex
    else:
        raise ValueError(f"Invalid tree node value: {f}")
    
    # Convert f_x to complex if discount is complex
    # if is_complex: f_x = to_complex(f_x)
    
    result = f_x
    for direction, subtree in tree[1]:
        sub = sum_tree(x, subtree, discount, semiring=semiring, maxplus_linear=maxplus_linear, channel_wise=channel_wise)
        if hasattr(subtree[0], 'activation_function_hack'):
            sub = subtree[0].activation_function_hack(sub)
        c_sum = ds_cumsum_XX(sub, direction, discount, semiring)
        if semiring in [Semiring.REAL, Semiring.MAXTIMES]:
            # Use complex multiplication if either operand is complex
            result = complex_multiply(result, c_sum)
        elif semiring == Semiring.MAXPLUS:
            # Use complex addition if either operand is complex
            result = result + c_sum
        else:
            raise ValueError(f"Invalid semiring: {semiring}")
    return result


def sum_trees(
    x: torch.Tensor, trees: list[Tree], discount: torch.Tensor, semiring: str = Semiring.REAL, 
    maxplus_linear: bool = False, cat: bool = False
) -> torch.Tensor:
    """
    x: (..., channels, height, width)
    trees: List of trees.  See sum_tree for more details.
    discount: (..., channels, height, width) discount tensor
    semiring: Semiring to use for the sum.  See Semiring for more details.
    cat: If True, concatenate the results along an existing dimension.

    Returns:
        (..., |trees|, height, width)
    """
    xs = []
    for tree in trees:
        tmp = sum_tree(x, tree, discount, semiring=semiring, maxplus_linear=maxplus_linear)
        xs.append(tmp)
    if cat:
        return torch.cat(xs, dim=-3)
    else:
        return torch.stack(xs, dim=-3)


def test_ds_cummax_E():
    X = torch.tensor([[[[1., 0.,  3.],
                        [4., -5., 0.]],
                       ]], device='cuda')
    discount = -2 * torch.ones_like(X)
    Y_N = ds_cummax_N(X, discount)
    Y_S = ds_cummax_S(X, discount)
    Y_W = ds_cummax_W(X, discount)
    Y_E = ds_cummax_E(X, discount)

    expected_Y_N=torch.tensor([[[[ 1.,  0.,  3.],
                                 [ 4., -2.,  1.]]]], device='cuda')
    expected_Y_S=torch.tensor([[[[ 2.,  0.,  3.],
                                 [ 4., -5.,  0.]]]], device='cuda')
    expected_Y_W=torch.tensor([[[[1., 0., 3.],
                                 [4., 2., 0.]]]], device='cuda')
    expected_Y_E=torch.tensor([[[[ 1.,  1.,  3.],
                                 [ 4., -2.,  0.]]]], device='cuda')

    torch.testing.assert_close(Y_N, expected_Y_N)
    torch.testing.assert_close(Y_S, expected_Y_S)
    torch.testing.assert_close(Y_W, expected_Y_W)
    torch.testing.assert_close(Y_E, expected_Y_E)

def test_ds_cumsum_E():
    X = torch.tensor([[[[1., 0.,  3.],
                        [4., -5., 0.]],
                       ]], device='cuda')
    discount = (1/2) * torch.ones_like(X)
    Y_N = ds_cumsum_N(X, discount)
    Y_S = ds_cumsum_S(X, discount)
    Y_W = ds_cumsum_W(X, discount)
    Y_E = ds_cumsum_E(X, discount)

    expected_Y_N=torch.tensor([[[[ 1.,  0.,  3.],
                                 [ 4.5, -5., 1.5]]]], device='cuda')
    expected_Y_S=torch.tensor([[[[ 3.,  -2.5,  3.],
                                 [ 4., -5.,    0.]]]], device='cuda')
    expected_Y_W=torch.tensor([[[[1., 0.5, 3.25],
                                 [4., -3., -1.5]]]], device='cuda')
    expected_Y_E=torch.tensor([[[[1.75, 1.5,  3.],
                                 [1.5, -5.,  0.]]]], device='cuda')

    torch.testing.assert_close(Y_N, expected_Y_N)
    torch.testing.assert_close(Y_S, expected_Y_S)
    torch.testing.assert_close(Y_W, expected_Y_W)
    torch.testing.assert_close(Y_E, expected_Y_E)

def test_ds_cummax_NW():
    torch.manual_seed(0)
    X = torch.randint(-10,10,(1,1,4,4), device='cuda')
    discount = -2 * torch.ones_like(X)
    Y_NW = ds_cummax_NW(X, discount)

    tmp = ds_cummax_W(X, discount)
    Y_NW_alt = ds_cummax_N(tmp, discount)
    torch.testing.assert_close(Y_NW, Y_NW_alt)

    # Not well-defined for non-constant discount.
    discount = torch.randint(-10,10,(1,1,4,4), device='cuda')
    Y_NW = ds_cummax_NW(X, discount)
    tmp = ds_cummax_W(X, discount)
    Y_NW_alt = ds_cummax_N(tmp, discount)
    assert not torch.allclose(Y_NW, Y_NW_alt)

def try_ds_sum_tree():
    """Test the discounted sum_tree function."""
    torch.manual_seed(0)
    X = torch.randn(1, 2, 3, 3, device='cuda')
    discount = 0.5 * torch.ones_like(X)
    
    # Simple tree: [1, [('N', [2, []])]]
    tree = [1.0, [('N', [2.0, []])]]
    
    result = sum_tree(X, tree, discount, semiring=Semiring.REAL)
    print(f"Discounted sum_tree result shape: {result.shape}")
    print(f"Discounted sum_tree result: {result}")
    
    # Test with maxplus semiring
    result_maxplus = sum_tree(X, tree, discount, semiring=Semiring.MAXPLUS)
    print(f"Discounted sum_tree (maxplus) result shape: {result_maxplus.shape}")
    print(f"Discounted sum_tree (maxplus) result: {result_maxplus}")

    # Test with acitvation function hack
    alpha_1 = torch.nn.Parameter(torch.randn(1,2, device='cuda'))
    alpha_2 = torch.nn.Parameter(torch.randn(1,2, device='cuda'))

    tree = [alpha_1, [(TreeDirection.W, [alpha_2, []])]]
    result = sum_tree(X, tree, discount, semiring=Semiring.REAL)
    print( 'result=', result, 'result.shape=', result.shape )
    assert not torch.allclose(result, torch.zeros_like(result))

    def activation(x):
        return 0. * x

    alpha_2.activation_function_hack = activation
    result = sum_tree(X, tree, discount, semiring=Semiring.REAL)
    torch.testing.assert_close(result, torch.zeros_like(result))
    # print( 'result=', result, 'result.shape=', result.shape )

def test_discount_identity():
    from src.models.sums import sum_tree as undiscounted_sum_tree
    torch.manual_seed(0)
    X = torch.randn(1, 2, 3, 3, device='cuda')
    discount = torch.ones_like(X)
    tree = [1.0, [(TreeDirection.W, [1.0, []])]]
    
    discounted_result = sum_tree(X, tree, discount, semiring=Semiring.REAL)
    undiscounted_result = undiscounted_sum_tree(X, tree, semiring=Semiring.REAL)
    torch.testing.assert_close(discounted_result, undiscounted_result)

    discount_no_spatial = torch.ones(1, 2, 1, 1, device='cuda')
    discounted_result_expanded = sum_tree(X, tree, discount_no_spatial, semiring=Semiring.REAL)
    undiscounted_result_no_spatial = undiscounted_sum_tree(X, tree, semiring=Semiring.REAL)
    assert not torch.allclose(discounted_result_expanded, undiscounted_result_no_spatial) # I think it reads discount from undefined memory regions

    # expan discount_no_spatial to have spatial dimensions
    expanded_discount_no_spatial = discount_no_spatial.expand(1, 2, 3, 3)
    discounted_result_expanded = sum_tree(X, tree, expanded_discount_no_spatial, semiring=Semiring.REAL)
    undiscounted_result_no_spatial = undiscounted_sum_tree(X, tree, semiring=Semiring.REAL)
    torch.testing.assert_close(discounted_result_expanded, undiscounted_result_no_spatial)

def test_batch_contamination():
    torch.manual_seed(0)
    x = torch.randn(2,8,28,28, device='cuda')
    x_zeroed = x.clone()
    x_zeroed[0] = x_zeroed[0] * 0.
    discount = torch.randn(2,8,28,28, device='cuda')
    discount_zeroed = discount.clone()
    discount_zeroed[0] = discount_zeroed[0] * 0.
    for f in [ ds_cumsum_S, ds_cumsum_W, ds_cumsum_E, ds_cumsum_N, ds_cumsum_NE, ds_cumsum_NW, ds_cumsum_SE, ds_cumsum_SW,
               ds_cummax_S, ds_cummax_W, ds_cummax_E, ds_cummax_N, ds_cummax_NE, ds_cummax_NW, ds_cummax_SE, ds_cummax_SW ]:
        print(f'f={f.__name__}')
        cs = f(x, discount)
        cs_zeroed = f(x_zeroed, discount_zeroed)
        torch.testing.assert_close(cs[1], cs_zeroed[1])

def smoke_test_complex():
    torch.manual_seed(0)
    X = torch.randn(1, 3, 5, 4, 2, device='cuda')
    discount = torch.ones_like(X)
    tree = [1.0, [(TreeDirection.W, [1.0, []])]]
    result = sum_tree(X, tree, discount, semiring=Semiring.REAL)
    assert result.shape == X.shape

    discount = torch.ones(1, 7, 5, 4, 2, device='cuda')
    alpha_1 = torch.nn.Parameter(torch.randn(7, 3, device='cuda'))
    alpha_2 = torch.nn.Parameter(torch.randn(7, 3, device='cuda'))
    tree = [alpha_1, [(TreeDirection.W, [alpha_2, []])]]
    X.requires_grad = True
    discount.requires_grad = True
    result = sum_tree(X, tree, discount, semiring=Semiring.REAL)

    dresult = torch.rand_like(result)
    result.backward(dresult)
    assert X.grad.shape == X.shape
    assert discount.grad.shape == discount.shape
    assert alpha_1.grad.shape == alpha_1.shape
    assert alpha_2.grad.shape == alpha_2.shape

if __name__ == "__main__":
    test_ds_cummax_E()
    test_ds_cummax_NW()
    test_ds_cumsum_E()
    try_ds_sum_tree()
    test_discount_identity()
    test_batch_contamination()
    smoke_test_complex()