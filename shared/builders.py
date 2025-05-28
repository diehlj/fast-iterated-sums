from typing import Generator

import sympy
import torch

from shared.defintions import Tree, TreeDirection, TreeType


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
        tree_type: type of tree to generate ('random', 'linear', 'linear_NW')

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

    elif tree_type == TreeType.LINEAR_NW:

        def cardinal_generator():
            while True:
                yield TreeDirection.NW

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


# XXX does this belong here?
def pretty_print_tree(tree, depth=0):
    value, children = tree
    indent = "    " * (depth + 1)  # 4 spaces per level

    # If value is callable, show that explicitly; otherwise just convert to string
    if depth == 0:
        if callable(value):
            node_str = "<function>"
        else:
            node_str = "<" + str(value)[:3] + ">"

        print(f"{node_str}")

    for direction, subtree in children:
        val_sub, _ = subtree
        if callable(val_sub):
            node_str = "<function>"
        else:
            node_str = "<" + str(val_sub)[:3] + ">"

        print(f"{indent}{direction} {node_str}")
        pretty_print_tree(subtree, depth + 1)


if __name__ == "__main__":

    def gen():
        while True:
            yield torch.rand(3)

    for t in create_forest(gen(), 10, 3, 0, tree_type="random"):
        print(t)
        pretty_print_tree(t)
    for t in create_forest(gen(), 10, 3, 0, tree_type="linear"):
        print(t)
        pretty_print_tree(t)
    for t in create_forest(gen(), 10, 3, 0, tree_type="linear_NW"):
        print(t)
        pretty_print_tree(t)
