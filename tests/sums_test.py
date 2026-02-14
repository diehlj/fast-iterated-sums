import torch
import torch.nn as nn

from src.models.sums import (
    Semiring,
    TreeDirection,
    cummax_NW,
    cummax_SW,
    cumsum_NW,
    cumsum_SW,
    sum_tree,
    sum_trees,
    Tree
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

def cummax_NW_manual(X: torch.Tensor) -> torch.Tensor:
    m, n = X.shape
    C = torch.zeros(m, n, dtype=X.dtype)
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                C[i, j] = X[i, j]
            elif i == 0:
                C[i, j] = max(C[i, j - 1], X[i, j])
            elif j == 0:
                C[i, j] = max(C[i - 1, j], X[i, j])
            else:
                C[i, j] = max(C[i - 1, j], C[i, j - 1], X[i, j]) # works, since max is idempotent
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

def cummax_SW_manual(X: torch.Tensor) -> torch.Tensor:
    m, n = X.shape
    C = torch.zeros(m, n, dtype=X.dtype)
    for i in range(m - 1, -1, -1):
        for j in range(n):
            if i == m - 1 and j == 0:
                C[i, j] = X[i, j]
            elif j == 0:
                C[i, j] = max(C[i + 1, j], X[i, j])
            elif i == m - 1:
                C[i, j] = max(C[i, j - 1], X[i, j])
            else:
                C[i, j] = max(C[i + 1, j], C[i, j - 1], X[i, j]) # works, since max is idempotent
    return C

def test_cumsum_XX():
    X = torch.tensor([[1, 2, 3], [4, 5, 6], [77, 8, 999]])
    assert torch.allclose(cumsum_NW(X), cumsum_NW_manual(X))
    assert torch.allclose(cumsum_SW(X), cumsum_SW_manual(X))

    X = torch.tensor([[1, 22, 3], [4, 5, 66], [77, 8, 999]])
    assert torch.allclose(
        cummax_NW(X), torch.tensor([[1, 22, 22], [4, 22, 66], [77, 77, 999]])
    )

    assert torch.allclose(cummax_NW(X), cummax_NW_manual(X))
    assert torch.allclose(cummax_SW(X), cummax_SW_manual(X))


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

#class TreeSummer(torch.nn.Module):
#    def __init__(self, trees: list[Tree]):
#        """
#        trees: List of trees.  See sum_tree for more details.
#        """
#        super(TreeSummer, self).__init__()
#        self.trees = trees
#
#    def forward(self, X: torch.Tensor) -> torch.Tensor:
#        """
#        x: (..., channels, height, width)
#
#        Returns:
#        (..., |self.trees|, height, width)
#        """
#        xs = []
#        for tree in self.trees:
#            tmp = sum_tree(X, tree)
#            xs.append(tmp)
#        return torch.stack(xs, dim=-3)
    
def test_bundled_sum():
    alpha_1 = torch.randn(6,3)
    alpha_2 = torch.randn(6,3)

    X = torch.randn(1, 3, 4, 4)

    tree_linear = [alpha_1, [(TreeDirection.NE, [alpha_2, []])]]
    out_bundled = sum_tree(X, tree_linear)

    #summer = TreeSummer( [alpha_1[i], [(TreeDirection.NE, [alpha_2[i], []])]] for i in range(6) )
    #out_separate = summer(X)
    #assert torch.allclose(out_bundled, out_separate)
    out_separate_2 = sum_trees(X, [ [alpha_1[i], [(TreeDirection.NE, [alpha_2[i], []])]] for i in range(6) ] )
    assert torch.allclose(out_bundled, out_separate_2)

    battery_out = sum_trees(X, [tree_linear, tree_linear], cat=True)
    assert battery_out.shape == (1,12,4,4)
    # TODO time this

def test_parameter_sharing():
    num_trees = 6
    in_channels = 2
    semiring = Semiring.REAL
    alpha_1  = nn.Parameter(torch.randn(num_trees//2,in_channels))
    alpha_2  = nn.Parameter(torch.randn(num_trees//2,in_channels))
    alpha_3  = nn.Parameter(torch.randn(num_trees//2,in_channels))
    alpha_1p = nn.Parameter(torch.randn(num_trees//2,in_channels))
    alpha_2p = nn.Parameter(torch.randn(num_trees//2,in_channels))
    alpha_3p = nn.Parameter(torch.randn(num_trees//2,in_channels))
    tree_linear  = [alpha_1,  [(TreeDirection.NE, [alpha_2,  [(TreeDirection.SW, [alpha_3,  []])]])]]
    tree_linearp = [alpha_1p, [(TreeDirection.NE, [alpha_2p, [(TreeDirection.SW, [alpha_3p, []])]])]]

    def _sum(x: torch.Tensor) -> torch.Tensor:
        return sum_trees(x, [tree_linear, tree_linearp], semiring=semiring, cat=True)
    sum = _sum
    X = torch.randn(1, in_channels, 4, 4)
    out = sum(X)
    assert out.shape == (1, num_trees, 4, 4)