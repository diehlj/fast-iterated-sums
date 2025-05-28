import torch

from shared.defintions import Semiring, Tree, TreeDirection


def cumsum_XX(x: torch.Tensor, XX: str, semiring: str = Semiring.REAL) -> torch.Tensor:
    if semiring == Semiring.REAL:
        if XX == TreeDirection.NW:
            return cumsum_NW(x)
        elif XX == TreeDirection.SW:
            return cumsum_SW(x)
        elif XX == TreeDirection.NE:
            return cumsum_NE(x)
        elif XX == TreeDirection.SE:
            return cumsum_SE(x)
        elif XX == TreeDirection.N:
            return cumsum_N(x)
        elif XX == TreeDirection.S:
            return cumsum_S(x)
        elif XX == TreeDirection.W:
            return cumsum_W(x)
        elif XX == TreeDirection.E:
            return cumsum_E(x)

    elif semiring == Semiring.MAXPLUS:
        if XX == TreeDirection.NW:
            return cummax_NW(x)
        elif XX == TreeDirection.SW:
            return cummax_SW(x)
        elif XX == TreeDirection.NE:
            return cummax_NE(x)
        elif XX == TreeDirection.SE:
            return cummax_SE(x)
        elif XX == TreeDirection.N:
            return cummax_N(x)
        elif XX == TreeDirection.S:
            return cummax_S(x)
        elif XX == TreeDirection.W:
            return cummax_W(x)
        elif XX == TreeDirection.E:
            return cummax_E(x)
    else:
        raise ValueError(
            f"Invalid semiring: {semiring}. Value must be one of {', '.join([sr for sr in Semiring])}"
        )


def sum_trees(
    x: torch.Tensor, trees: list[Tree], semiring: str = Semiring.REAL
) -> torch.Tensor:
    """
    x: (..., channels, height, width)
    trees: List of trees.  See sum_tree for more details.
    semiring: Semiring to use for the sum.  See Semiring for more details.

    Returns:
        (..., |trees|, height, width)
    """
    xs = []
    for tree in trees:
        tmp = sum_tree(x, tree, semiring=semiring)
        xs.append(tmp)
    return torch.stack(xs, dim=-3)


def sum_tree(
    x: torch.Tensor, tree: Tree, semiring: str = Semiring.REAL
) -> torch.Tensor:
    """
    Calculate the sum corresponding to a corner tree.

    Args:
        x: (..., channels, height, width)
        tree: [value, [(direction, subtree), ...]]
                'subtree' is of same form, or [value, []] for a leaf.
                'direction' is a TreeDirection (one of N, NE, E, SE, S, SW, W, NW).
                'value' can be a scalar, a tensor of shape (channels,), or a function that takes x as input
                        (..., channels, height, width) and returns (..., height, width).
        semiring: Semiring to use for the sum.  See Semiring for more details.

    Returns:
        (..., height, width)
    """
    # print('tree', tree)
    f = tree[0]
    # channels = x.shape[-3]
    if callable(f):
        f_x = f(x)
    elif isinstance(f, (int, float)) or (torch.is_tensor(f) and f.numel() == 1):
        f_x = f * x
    elif torch.is_tensor(f):
        # TODO: maybe add bias ?
        f_x = torch.einsum("...cwh, c->...wh", x, f)
    else:
        raise ValueError(f"Invalid tree node value: {f}")
    result = f_x
    for direction, subtree in tree[1]:
        sub = sum_tree(x, subtree)
        c_sum = cumsum_XX(sub, direction, semiring)
        if semiring == Semiring.REAL:
            result *= c_sum
        elif semiring == Semiring.MAXPLUS:
            result += c_sum
        else:
            raise ValueError(f"Invalid semiring: {semiring}")
    return result

def cumsum_NW(X: torch.Tensor) -> torch.Tensor:
    """
    X: (..., height, width)

    Returns:
    cumsum_NW(X)[i,j] = sum_{(k,l) to the NW of (i,j)} X[k,l]

    In the usual 2D array indexing, this is equivalent to:

        cumsum_NW(X)[i,j] = sum_{k <= i, l <= j} X[k,l]
    """
    tmp = torch.cumsum(X, dim=-2)
    return torch.cumsum(tmp, dim=-1)


def cummax_NW(X: torch.Tensor) -> torch.Tensor:
    """
    X: (..., height, width)

    Returns:
    cummax_NW(X)[i,j] = max_{(k,l) to the NW of (i,j)} X[k,l]

    In the usual 2D array indexing, this is equivalent to:

        cummax_NW(X)[i,j] = max_{k <= i, l <= j} X[k,l]
    """
    tmp = torch.cummax(X, dim=-2)[0]
    return torch.cummax(tmp, dim=-1)[0]


def cumsum_N(X: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(X, dim=-2)


def cumsum_S(X: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(X, [-2]), dim=-2), [-2])


def cumsum_W(X: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(X, dim=-1)


def cumsum_E(X: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(X, [-1]), dim=-1), [-1])


def cummax_N(X: torch.Tensor) -> torch.Tensor:
    return torch.cummax(X, dim=-2)[0]


def cummax_S(X: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.cummax(torch.flip(X, [-2]), dim=-2)[0], [-2])


def cummax_W(X: torch.Tensor) -> torch.Tensor:
    return torch.cummax(X, dim=-1)[0]


def cummax_E(X: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.cummax(torch.flip(X, [-1]), dim=-1)[0], [-1])


def cummax_SW(X: torch.Tensor) -> torch.Tensor:
    tmp = torch.flip(X, [-2])
    tmp = cummax_NW(tmp)
    return torch.flip(tmp, [-2])


def cummax_NE(X: torch.Tensor) -> torch.Tensor:
    tmp = torch.flip(X, [-1])
    tmp = cummax_NW(tmp)
    return torch.flip(tmp, [-1])


def cummax_SE(X: torch.Tensor) -> torch.Tensor:
    tmp = torch.flip(X, [-2, -1])
    tmp = cummax_NW(tmp)
    return torch.flip(tmp, [-2, -1])


def cumsum_SW(X: torch.Tensor) -> torch.Tensor:
    tmp = torch.flip(X, [-2])
    tmp = cumsum_NW(tmp)
    return torch.flip(tmp, [-2])


def cumsum_NE(X: torch.Tensor) -> torch.Tensor:
    tmp = torch.flip(X, [-1])
    tmp = cumsum_NW(tmp)
    return torch.flip(tmp, [-1])


def cumsum_SE(X: torch.Tensor) -> torch.Tensor:
    tmp = torch.flip(X, [-2, -1])
    tmp = cumsum_NW(tmp)
    return torch.flip(tmp, [-2, -1])


# https://github.com/proger/accelerated-scan
# XXX cumsum/cummax seem already optimized
# import accelerated_scan
# def benchmark_cumsum():
#    X = torch.randn(1_000_000)
#    import time
#    start = time.time()
#    for _ in range(10):
#        torch.cumsum(X, dim=-1)
#    print("torch.cumsum", time.time()-start)
#    # start = time.time()
#    start = time.time()
#    for _ in range(10):
#        torch.cummax(X, dim=-1)
#    print("torch.cummax", time.time()-start)
#
#    start = time.time()
#    for _ in range(10):
#        accelerated_scan.warp.scan
#    print("accelerated_scan.cumsum", time.time()-start)

# TODO:
# - scaling
#    - [ ] look at Krieg's scaling
# - non-linear f
# - normalization
# - visualization of size of results
# - imaginary exponential
#   https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf
# - cifar10
# - multidim (pointwise) sums (i.e. heads)
#
# LATER
# - learn shape of trees (discrete learning)