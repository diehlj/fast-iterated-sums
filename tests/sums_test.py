import torch

from shared.sums import (
    Semiring,
    TreeDirection,
    cummax_NW,
    cumsum_NW,
    cumsum_SW,
    sum_tree,
    sum_trees,
)


def cumsum_NW_manual(X: torch.Tensor) -> torch.Tensor:
    m, n = X.shape
    C = torch.zeros(m, n, dtype=X.dtype)
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                C[i, j] = X[i, j]
            elif i == 0:
                C[i, j] = C[i, j - 1] + X[i, j]
            elif j == 0:
                C[i, j] = C[i - 1, j] + X[i, j]
            else:
                C[i, j] = C[i - 1, j] + C[i, j - 1] - C[i - 1, j - 1] + X[i, j]
    return C


def cumsum_SW_manual(X: torch.Tensor) -> torch.Tensor:
    m, n = X.shape
    C = torch.zeros(m, n, dtype=X.dtype)
    for i in range(m - 1, -1, -1):
        for j in range(n):
            if i == m - 1 and j == 0:
                C[i, j] = X[i, j]
            elif j == 0:
                C[i, j] = C[i + 1, j] + X[i, j]
            elif i == m - 1:
                C[i, j] = C[i, j - 1] + X[i, j]
            else:
                C[i, j] = C[i + 1, j] + C[i, j - 1] - C[i + 1, j - 1] + X[i, j]
    return C


def test_cumsum_XX():
    X = torch.tensor([[1, 2, 3], [4, 5, 6], [77, 8, 999]])
    C = cumsum_NW(X)
    C_manual = cumsum_NW_manual(X)
    assert torch.allclose(C, C_manual)
    assert torch.allclose(cumsum_SW(X), cumsum_SW_manual(X))

    X = torch.tensor([[1, 22, 3], [4, 5, 66], [77, 8, 999]])
    assert torch.allclose(
        cummax_NW(X), torch.tensor([[1, 22, 22], [4, 22, 66], [77, 77, 999]])
    )


def test_sum_tree():
    tree_dot = [1.0, []]
    tree_linear = [1.0, [(TreeDirection.NE, [1.0, []])]]
    tree_cherry = [1.0, [(TreeDirection.SE, [1.0, []]), (TreeDirection.SW, [1.0, []])]]

    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    Y = sum_tree(X, tree_linear, semiring=Semiring.MAXPLUS)
    assert torch.allclose(Y, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))

    tree_linear_nl = [
        lambda x: x**2,
        [(TreeDirection.SE, [lambda x: (x - 2) ** 3, []])],
    ]
    Y_linear = sum_tree(X, tree_linear_nl)
    assert torch.allclose(Y_linear, torch.tensor([[8.0, 32.0], [81.0, 128.0]]))

    X = torch.ones(2, 2)
    expected = torch.tensor([[8.0, 8.0], [2.0, 2.0]])
    assert torch.allclose(sum_tree(X, tree_cherry), expected)

    X = torch.ones(3, 3)
    Y_linear = sum_tree(X, tree_linear)
    assert torch.allclose(
        Y_linear, torch.tensor([[3.0, 2.0, 1.0], [6.0, 4.0, 2.0], [9.0, 6.0, 3.0]])
    )
    Y_dot = sum_tree(X, tree_dot)
    assert torch.allclose(X, Y_dot)

    X = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    f = lambda x: torch.einsum("...cwh, c->...wh", x, torch.tensor([3.0, 0.0, -5.0]))
    tree_dot = [f, []]
    Y_dot = sum_tree(X, tree_dot)
    assert torch.allclose(Y_dot, torch.tensor([[-42.0, -44.0], [-46.0, -48.0]]))


def test_tree_sums():
    torch.manual_seed(0)
    H = W = 32
    x = torch.randn(1, 3, H, W)  # (batch_size, channels, height, width)

    tree_linear_simple = [
        torch.tensor([1.0, 0.0, 0.0]),
        [(TreeDirection.NW, [torch.tensor([1.0, 0.0, 0.0]), []])],
    ]
    result_simple = sum_tree(x, tree_linear_simple)
    # The result of sum_tree is the 'derivative' of the actual iterated sum.
    # To get the actual iterated sum, we need to sum one more time.
    final_sum = torch.sum(result_simple)

    by_hand_inner = torch.cumsum(torch.cumsum(x[..., 0, :, :], dim=-1), dim=-2)
    by_hand_outer = torch.sum(x[..., 0, :, :] * by_hand_inner)
    assert torch.allclose(final_sum, by_hand_outer)

    # Tree with weights (which are contracted)
    tree_linear_with_weights = [
        torch.tensor([0.1, 0.5, 0.8]),
        [(TreeDirection.NE, [torch.tensor([2.4, 0.1, -0.3]), []])],
    ]

    # Tree with functions that do the same
    def alpha_1(x):
        return torch.einsum("...cwh, c->...wh", x, torch.tensor([0.1, 0.5, 0.8]))

    def alpha_2(x):
        return torch.einsum("...cwh, c->...wh", x, torch.tensor([2.4, 0.1, -0.3]))

    tree_linear_with_fns = [alpha_1, [(TreeDirection.NE, [alpha_2, []])]]

    # Tree with polynomial functions
    def beta_1(x):
        return x[..., 0, :, :] ** 3 - x[..., 1, :, :] ** 2

    def beta_2(x):
        return torch.sin(x[..., 2, :, :]) * x[..., 0, :, :]

    tree_linear_with_fns_poly = [beta_1, [(TreeDirection.NE, [beta_2, []])]]

    # Compute sum for a single tree
    result = sum_tree(x, tree_linear_with_weights)
    result_ = sum_tree(x, tree_linear_with_fns)
    # print(result.shape)  # Output: torch.Size([1, 32, 32])
    assert result.shape == (1, H, W)
    assert torch.allclose(result, result_)

    # Using the max-plus semiring
    result = sum_tree(x, tree_linear_with_weights, semiring=Semiring.MAXPLUS)
    result_ = sum_tree(x, tree_linear_with_fns, semiring=Semiring.MAXPLUS)
    # print(result.shape)  # Output: torch.Size([1, 32, 32])
    assert result.shape == (1, H, W)
    assert torch.allclose(result, result_)

    result_poly = sum_tree(x, tree_linear_with_fns_poly)
    # print(result_poly.shape)  # Output: torch.Size([1, 32, 5])
    assert result_poly.shape == (1, H, W)

    # Another tree with weights
    tree_cherry = [
        torch.tensor([-3.0, 0.0, 1.0]),
        [
            (TreeDirection.SE, [torch.tensor([1.0, 2.0, -1.0]), []]),
            (TreeDirection.SW, [torch.tensor([3.0, 6.0, -1.0]), []]),
        ],
    ]
    # Compute sums for multiple trees
    trees = [tree_cherry, tree_linear_with_weights, tree_linear_with_fns_poly]
    result_multiple = sum_trees(x, trees)
    # print(result_multiple.shape)  # Output: torch.Size([1, 3, 32, 32])
    assert result_multiple.shape == (1, 3, H, W)
