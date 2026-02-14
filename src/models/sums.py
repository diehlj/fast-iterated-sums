from enum import IntEnum, StrEnum, auto
from types import MappingProxyType
from typing import Generator
import sympy
import torch

class TreeType(StrEnum):
    RANDOM = auto()
    LINEAR = auto()
    LINEAR_NE = "linear_NE"

class Semiring(StrEnum):
    REAL = auto()
    MAXPLUS = auto()
    MAXTIMES = auto()

class SelectMode(StrEnum):
    ALL_RANDOM = auto()
    ALL_TRAIN = auto()

class TreeDirection(StrEnum):
    NE = "NE"
    SE = "SE"
    NW = "NW"
    SW = "SW"
    N = "N"
    S = "S"
    W = "W"
    E = "E"

type Tree = list[float | int | torch.Tensor | list[tuple[str, list]]]

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

    elif semiring in [Semiring.MAXPLUS, Semiring.MAXTIMES]:
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
    x: torch.Tensor, trees: list[Tree], semiring: str = Semiring.REAL, maxplus_linear: bool = False,
    cat: bool = False
) -> torch.Tensor:
    """
    x: (..., channels, height, width)
    trees: List of trees.  See sum_tree for more details.
    semiring: Semiring to use for the sum.  See Semiring for more details.
    cat: If True, concatenate the results along an existing dimension.

    Returns:
        (..., |trees|, height, width)
    """
    xs = []
    for tree in trees:
        tmp = sum_tree(x, tree, semiring=semiring, maxplus_linear=maxplus_linear)
        xs.append(tmp)
    if cat:
        return torch.cat(xs, dim=-3)
    else:
        return torch.stack(xs, dim=-3)


def linear_wh_channelwise(x, f, maxplus_linear=False): # XXX I think it should be _hw
    '''
    Channel-wise linear map. Mostly useful when using the WRONG semiring, e.g.

        sum a x_s b x_t = ab sum x_s x_t -> not useful
        sum (a + x_s)(b + x_t)           -> maybe useful

        max (a + x_s + b + x_t) = a + b + max x_s + x_t  -> not useful
        max (a x_s + b x_t)                              -> maybe useful
    '''
    if len(f.shape) == 1:
        if maxplus_linear:
            f_expanded = f.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, channels, 1, 1]
            tmp = x + f_expanded                                    # broadcasting
            return tmp
        else:
            f_x = torch.einsum("...cwh, c->...cwh", x, f) # XXX I think it should be _hw
    else:
        raise ValueError("Channel-wise linear does not support f with shape (out_channels, in_channels)")
    return f_x


def linear_wh(x, f, maxplus_linear=False, is_complex=False): # XXX I think it should be _hw
    if is_complex: assert not maxplus_linear, "Maxplus linear is not supported for complex tensors"
    if len(f.shape) == 1:
        if maxplus_linear:
            f_expanded = f.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, channels, 1, 1]
            tmp = x + f_expanded             # broadcasting
            f_x, _ = torch.max(tmp, dim=-3)  # max over channels
        else:
            if is_complex:
                f_x = torch.einsum("...cwhi, c->...cwhi", x, f)
            else:
                f_x = torch.einsum("...cwh, c->...wh", x, f) # XXX I think it should be _hw
    else:
        if maxplus_linear:
            f_expanded = f.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, out_channels, in_channels, 1, 1]
            x_expanded = x.unsqueeze(-4)    # [..., 1, in_channels, height, width]
            tmp = x_expanded + f_expanded   # broadcasting
            f_x, _ = torch.max(tmp, dim=-3) # max over in_channels
        else:
            if is_complex:
                f_x = torch.einsum("...cwhi, oc->...owhi", x, f)
            else:
                f_x = torch.einsum("...cwh, oc->...owh", x, f)
    return f_x

def sum_tree(
    x: torch.Tensor, tree: Tree, semiring: str = Semiring.REAL, maxplus_linear: bool = False,
    channel_wise: bool = False,
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
        maxplus_linear: If True, use tropical linear map for the node values.

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
        if channel_wise:
            f_x = linear_wh_channelwise(x, f, maxplus_linear=maxplus_linear)
        else:
            f_x = linear_wh(x, f, maxplus_linear=maxplus_linear)
    else:
        raise ValueError(f"Invalid tree node value: {f}")
    result = f_x
    for direction, subtree in tree[1]:
        sub = sum_tree(x, subtree, semiring=semiring, maxplus_linear=maxplus_linear, channel_wise=channel_wise) # FIXED a bug here: semiring was NOT passed
        if hasattr(subtree[0], 'activation_function_hack'):
            sub = subtree[0].activation_function_hack(sub)
        c_sum = cumsum_XX(sub, direction, semiring)
        if semiring in [Semiring.REAL, Semiring.MAXTIMES]:
            # result *= c_sum
            result = result * c_sum
        elif semiring == Semiring.MAXPLUS:
            # result += c_sum
            result = result + c_sum
        else:
            raise ValueError(f"Invalid semiring: {semiring}")
    return result




def cumsum_N(X: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(X, dim=-2)

def cumsum_E(X: torch.Tensor) -> torch.Tensor:
    return torch.flip(cumsum_W(torch.flip(X, [-1])), [-1])

def cumsum_S(X: torch.Tensor) -> torch.Tensor:
    return torch.flip(cumsum_N(torch.flip(X, [-2])), [-2])

def cumsum_W(X: torch.Tensor) -> torch.Tensor:
    return torch.transpose(torch.cumsum(torch.transpose(X, -2, -1), dim=-2), -2, -1)

def cumsum_NW(X: torch.Tensor) -> torch.Tensor:
    """
    X: (..., height, width)

    Returns:
    cumsum_NW(X)[i,j] = sum_{(k,l) to the NW of (i,j)} X[k,l]

    In the usual 2D array indexing, this is equivalent to:

        cumsum_NW(X)[i,j] = sum_{k <= i, l <= j} X[k,l]
    """
    tmp = cumsum_N(X)
    return cumsum_W(tmp)

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



def cummax_N(X: torch.Tensor) -> torch.Tensor:
    return torch.cummax(X, dim=-2)[0]

def cummax_S(X: torch.Tensor) -> torch.Tensor:
    return torch.flip(cummax_N(torch.flip(X, [-2])), [-2])

def cummax_W(X: torch.Tensor) -> torch.Tensor:
    return torch.transpose(cummax_N(torch.transpose(X, -2, -1)), -2, -1)

def cummax_E(X: torch.Tensor) -> torch.Tensor:
    return torch.flip(cummax_W(torch.flip(X, [-1])), [-1])

def cummax_NW(X: torch.Tensor) -> torch.Tensor:
    """
    X: (..., height, width)

    Returns:
    cummax_NW(X)[i,j] = max_{(k,l) to the NW of (i,j)} X[k,l]

    In the usual 2D array indexing, this is equivalent to:

        cummax_NW(X)[i,j] = max_{k <= i, l <= j} X[k,l]
    """
    tmp = cummax_N(X)
    return cummax_W(tmp)

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



def random_tree(
    alpha_generator: Generator,
    num_nodes: int,
) -> Tree:
    """Generate a random tree with num_nodes nodes.

    Args:
        alpha_generator: generator that yields values for the nodes
        num_nodes: number of nodes in the tree

    Returns:
        Tree
    """
    CARDINALS = [td for td in TreeDirection]

    if num_nodes == 1:
        return [next(alpha_generator), []]
    elif num_nodes > 1:
        all_partitions = list(
            sympy.utilities.iterables.ordered_partitions(num_nodes - 1)
        )
        random_partition = all_partitions[torch.randint(len(all_partitions), (1,))]
        children = [random_tree(alpha_generator, p) for p in random_partition]
        directions = [
            CARDINALS[torch.randint(len(CARDINALS), (1,))] for _ in random_partition
        ]
        return [next(alpha_generator), list(zip(directions, children))]
    else:
        raise ValueError("num_nodes must be positive")


def create_forest(
    alpha_generator: Generator,
    num_trees: int,
    num_nodes: int,
    seed: int | None = None,
    tree_type: str = TreeType.RANDOM,
) -> list[Tree]:
    """Generate a list of trees.

    Args:
        alpha_generator: generator that yields values for the nodes
        num_trees: number of trees to generate
        num_nodes: number of nodes in each tree
        seed: random seed for reproducibility
        tree_type: type of tree to generate ('random', 'linear', 'linear_NE')

    Returns:
        list: of trees
    """
    if seed is not None and isinstance(seed, int):
        torch.manual_seed(seed=seed)

    if num_trees <= 0 or not isinstance(num_trees, int):
        raise ValueError(
            f"num_trees must be integer and greater than 0; but {num_trees} was provided"
        )

    if tree_type == TreeType.RANDOM:
        return [
            random_tree(alpha_generator=alpha_generator, num_nodes=num_nodes)
            for _ in range(num_trees)
        ]

    elif tree_type == TreeType.LINEAR:

        def cardinal_generator():
            CARDINALS = [td for td in TreeDirection]
            while True:
                yield CARDINALS[torch.randint(len(CARDINALS), (1,))]

        return [
            linear_tree(num_nodes, alpha_generator, cardinal_generator())
            for _ in range(num_trees)
        ]

    elif tree_type == TreeType.LINEAR_NE:

        def cardinal_generator():
            while True:
                yield TreeDirection.NE

        return [
            linear_tree(num_nodes, alpha_generator, cardinal_generator())
            for _ in range(num_trees)
        ]

    else:
        raise ValueError(
            f"tree_type must be one of {', '.join(TreeType)}; but {tree_type} was provided"
        )


def linear_tree(n, alpha_generator, cardinal_generator):
    if n == 0:
        return [next(alpha_generator), []]
    else:
        return [
            next(alpha_generator),
            [
                (
                    next(cardinal_generator),
                    linear_tree(n - 1, alpha_generator, cardinal_generator),
                )
            ],
        ]

def pretty_str_tree_one_line(tree):
    value, children = tree

    # If value is callable, show that explicitly; otherwise just convert to string
    if callable(value):
        node_str = "<function>"
    else:
        node_str = "<" + str(value)[:3] + ">"

    if not children:
        return node_str
    else:
        children_strs = []
        for direction, subtree in children:
            children_strs.append(f"{direction} {pretty_str_tree_one_line(subtree)}")
        return f"{node_str} -> ({', '.join(children_strs)})"

def param_fingerprint(p):
   import hashlib
   d = p.detach().cpu().numpy()
   h = hashlib.sha256(d.tobytes())
   h.update(str(p.shape).encode())
   return h.hexdigest()[:16]

def pretty_str_tree(tree, depth=0):
    value, children = tree
    indent = "    " * (depth + 1)  # 4 spaces per level

    def get_node_str(value):
        if callable(value):
            return "<function>"
        elif isinstance(value, torch.nn.Parameter):
            return f"<{value.shape} {tuple(value.shape)} {param_fingerprint(value)}>"
        else:
            return "<" + str(value)[:3] + ">"

    # If value is callable, show that explicitly; otherwise just convert to string
    if depth == 0:
        node_str = get_node_str(value)
        result = f"{node_str}\n"
    else:
        result = ""

    for direction, subtree in children:
        val_sub, _ = subtree
        node_str = get_node_str(val_sub)

        result += f"{indent}{direction} {node_str}\n"
        result += pretty_str_tree(subtree, depth + 1)
    return result

def pretty_print_tree(tree, depth=0):
    print( pretty_str_tree(tree, depth) )
    #value, children = tree
    #indent = "    " * (depth + 1)  # 4 spaces per level

    ## If value is callable, show that explicitly; otherwise just convert to string
    #if depth == 0:
    #    if callable(value):
    #        node_str = "<function>"
    #    else:
    #        node_str = "<" + str(value)[:3] + ">"

    #    print(f"{node_str}")

    #for direction, subtree in children:
    #    val_sub, _ = subtree
    #    if callable(val_sub):
    #        node_str = "<function>"
    #    else:
    #        node_str = "<" + str(val_sub)[:3] + ">"

    #    print(f"{indent}{direction} {node_str}")
    #    pretty_print_tree(subtree, depth + 1)

def all_trees(num_nodes: int) -> list[Tree]:
    """Generate all possible trees with num_nodes nodes.

    Args:
        num_nodes: number of nodes in the tree

    Returns:
        list: of all possible trees
    """
    if num_nodes <= 0 or not isinstance(num_nodes, int):
        raise ValueError(
            f"num_nodes must be integer and greater than 0; but {num_nodes} was provided"
        )

    CARDINALS = [td for td in TreeDirection]

    if num_nodes == 1:
        return [[0, []]]  # use 0 as a placeholder for the node value
    else:
        all_trees_list = []
        all_partitions = list(sympy.utilities.iterables.ordered_partitions(num_nodes - 1))
        for partition in all_partitions:
            subtrees_lists = [all_trees(p) for p in partition]
            from itertools import product

            for children_combination in product(*subtrees_lists):
                from itertools import product

                for directions in product(CARDINALS, repeat=len(partition)):
                    children = list(zip(directions, children_combination))
                    all_trees_list.append([0, children])  # use 0 as a placeholder
        return all_trees_list

def n_vertices(tree):
    """Count the number of vertices in a tree.

    Args:
        tree: Tree

    Returns:
        int: number of vertices
    """
    count = 1  # count the root
    for _, subtree in tree[1]:
        count += n_vertices(subtree)
    return count

def test_linear_wh():
    torch.manual_seed(0)
    X = torch.randint(0,103, (1, 3, 2, 2)).float()
    alpha = torch.randint(0,10,(3,)).float()
    Y = linear_wh(X, alpha, maxplus_linear=True)
    expected = -torch.ones(1,2,2) * torch.inf
    for i in range(2):
        for j in range(2):
            for c in range(3):
                expected[0,i,j] = torch.maximum(expected[0,i,j], X[0,c,i,j] + alpha[c])
    assert torch.allclose(Y, expected)

    X = torch.randint(0,103, (1, 3, 2, 2)).float()
    alpha = torch.tensor([10., -21., 3.]).unsqueeze(0)
    Y = linear_wh(X, alpha, maxplus_linear=True)
    expected = -torch.ones(1,2,2) * torch.inf
    for i in range(2):
        for j in range(2):
            for c in range(3):
                expected[0,i,j] = torch.maximum(expected[0,i,j], X[0,c,i,j] + alpha[0,c])
    assert torch.allclose(Y, expected)

    # channel_wise
    X = torch.randint(0,103, (1, 3, 2, 2)).float()
    alpha = torch.tensor([1.0, 2.0, 3.0])
    ALPHA = torch.diag(alpha)
    Y    = linear_wh(X, ALPHA, maxplus_linear=False)
    Y_cw = linear_wh_channelwise(X, alpha, maxplus_linear=False)
    assert torch.allclose(Y, Y_cw)

    alpha = torch.tensor([1., -20., 3.])
    ALPHA = torch.diag(alpha)
    ALPHA[ALPHA == 0] = -torch.inf
    Y    = linear_wh(X, ALPHA, maxplus_linear=True)
    Y_cw = linear_wh_channelwise(X, alpha, maxplus_linear=True)
    assert torch.allclose(Y, Y_cw)


def test_batch_contamination():
    x = torch.randn(2,8,28,28)
    x_zeroed = x.clone()
    x_zeroed[0] = x_zeroed[0] * 0.
    for f in [ cumsum_S, cumsum_W, cumsum_E, cumsum_N, cumsum_NE, cumsum_NW, cumsum_SE, cumsum_SW,
               cummax_S, cummax_W, cummax_E, cummax_N, cummax_NE, cummax_NW, cummax_SE, cummax_SW]:
        cs = f(x)
        cs_zeroed = f(x_zeroed)
        torch.testing.assert_close(cs[1], cs_zeroed[1])

def show_some_trees():
    def gen():
        while True:
            yield torch.rand(3)

    for t in create_forest(gen(), 10, 3, 0, tree_type="random"):
        print(t)
        pretty_print_tree(t)
    for t in create_forest(gen(), 10, 3, 0, tree_type="linear"):
        print(t)
        pretty_print_tree(t)
    for t in create_forest(gen(), 10, 3, 0, tree_type="linear_NE"):
        print(t)
        pretty_print_tree(t)

    print()
    print('Testing all_trees(2):')
    for i,t in enumerate(all_trees(2)):
        assert n_vertices(t) == 2
        print(f'{i}: {pretty_str_tree_one_line(t)}')

    print('len(all_trees(2)) =', len(all_trees(2)))

    print()
    print('Testing all_trees(3):')
    for i, t in enumerate(all_trees(3)):
        assert n_vertices(t) == 3
        print(f'{i}: {pretty_str_tree_one_line(t)}')
        # pretty_print_tree(t)
        # print( pretty_str_tree_one_line(t) )

    print('len(all_trees(3)) =', len(all_trees(3)))

    num_trees = len(all_trees(3))


    for i in [450, 475]:
        print(f'{i}: {pretty_str_tree_one_line(all_trees(3)[i % num_trees])}')
        pretty_print_tree(all_trees(3)[i % num_trees])

    #print()
    #print()
    #print('Testing all_trees(4):')
    #for t in all_trees(4):
    #    assert n_vertices(t) == 4
    #    print(t)
    #    pretty_print_tree(t)

    #print('len(all_trees(4)) =', len(all_trees(4)))

if __name__ == "__main__":
    show_some_trees()
    # test_linear_wh()
    # test_batch_contamination()