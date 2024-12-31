import random
import numpy as np


def get_perceptron_layers(
    n_layers_grid: list[int],
    layer_size_grid: list[int],
    random_seed: int | None = None,
) -> list[tuple]:
    """
    Construct list of randomly sampled multilayer perceptron
    configuration tuples.

    Each tuple is randomly constructed given a grid of layer
    counts and a grid of layer sizes. A single tuple is just
    a sequence of layer sizes, eg. (10, 20, 60, 20, 10), for
    some diamond shaped perceptron.

    Parameters
    ----------
    n_layers_grid :
        List of potential layer counts determining how many
        perceptron layers there can be in a configuration tuple.
    layer_size_grid :
        List of potential perceptron layer sizes from which
        to construct a configuration tuple.
    random_seed :
        Random seed.

    Returns
    -------
    layer_tuples :
        Collection of tuples, each of which contains the layer sizes
        determining the architecture of a multilayer perceptron.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    layer_tuples = []
    # Hard coded:
    discretization = 1000
    for _ in range(discretization):
        tuple_len = random.choice(n_layers_grid)
        layer_tuple = ()
        for _ in range(tuple_len):
            layer_tuple = layer_tuple + (random.choice(layer_size_grid),)
        layer_tuples.append(layer_tuple)

    return layer_tuples


def q10(x):
    return x.quantile(0.1)


def q90(x):
    return x.quantile(0.9)
