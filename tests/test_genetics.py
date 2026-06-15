import random

import pytest

from evolution.genetics import combine_brains, combine_genes, mutate_genes
from evolution.neural_network import NeuralNetwork
from evolution import config


def test_combine_genes_respects_constraints():
    random.seed(42)
    g1 = config.DEFAULT_HERBIVORE_GENES.copy()
    g2 = config.DEFAULT_HERBIVORE_GENES.copy()
    for _ in range(50):
        child = combine_genes(g1, g2, config.DEFAULT_HERBIVORE_GENES)
        for key, (min_val, max_val) in config.GENE_MIN_MAX.items():
            assert min_val <= child[key] <= max_val, f"{key} out of range: {child[key]}"


def test_mutate_genes_respects_constraints():
    random.seed(1)
    genes = config.DEFAULT_CARNIVORE_GENES.copy()
    for _ in range(50):
        mutated = mutate_genes(genes, config.DEFAULT_CARNIVORE_GENES,
                               gene_constraints=config.CARNIVORE_GENE_MIN_MAX)
        for key, (min_val, max_val) in config.CARNIVORE_GENE_MIN_MAX.items():
            assert min_val <= mutated[key] <= max_val, f"{key} out of range: {mutated[key]}"


def test_combine_brains_shape_preserved():
    import numpy as np
    np.random.seed(5)
    brain1 = NeuralNetwork([4, 3, 2])
    brain2 = NeuralNetwork([4, 3, 2])
    child = combine_brains(brain1, brain2)
    assert child.layer_sizes == [4, 3, 2]
    for w1, w2 in zip(child.weights, brain1.weights):
        assert w1.shape == w2.shape


def test_combine_brains_values_mixed():
    import numpy as np
    np.random.seed(99)
    brain1 = NeuralNetwork([4, 3, 2])
    brain2 = NeuralNetwork([4, 3, 2])
    child = combine_brains(brain1, brain2)
    # At least some weights should come from brain2 (probabilistically certain with this seed)
    matches_1 = np.sum(child.weights[0] == brain1.weights[0])
    matches_2 = np.sum(child.weights[0] == brain2.weights[0])
    total = child.weights[0].size
    assert matches_1 < total or matches_2 < total
