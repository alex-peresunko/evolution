import random

import pytest

from evolution import config
from evolution.brain.network import NeuralNetwork
from evolution.genetics import combine_brains, combine_genes, mutate_genes
from evolution.species import CARNIVORE, HERBIVORE


def test_combine_genes_respects_constraints():
    random.seed(42)
    g1 = config.DEFAULT_HERBIVORE_GENES.copy()
    g2 = config.DEFAULT_HERBIVORE_GENES.copy()
    for _ in range(50):
        child = combine_genes(g1, g2, HERBIVORE)
        for key, (min_val, max_val) in config.GENE_MIN_MAX.items():
            assert min_val <= child[key] <= max_val, f"{key} out of range: {child[key]}"


def test_mutate_genes_respects_constraints():
    random.seed(1)
    genes = config.DEFAULT_CARNIVORE_GENES.copy()
    for _ in range(50):
        mutated = mutate_genes(genes, CARNIVORE)
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


def test_mutate_brain_uses_the_creatures_own_rates():
    import numpy as np

    from evolution.genetics import mutate_brain
    brain = NeuralNetwork([4, 3, 2])
    frozen = brain.copy()
    mutate_brain(frozen, {"mutation_rate": 0.0, "mutation_amount": 0.5})
    for a, b in zip(frozen.weights, brain.weights):
        np.testing.assert_array_equal(a, b)
    wild = brain.copy()
    mutate_brain(wild, {"mutation_rate": 1.0, "mutation_amount": 0.5})
    assert all((a != b).all() for a, b in zip(wild.weights, brain.weights))


def test_mutation_genes_are_inherited_and_clamped():
    random.seed(3)
    lo, hi = config.GENE_MIN_MAX["mutation_rate"]
    g1 = {**config.DEFAULT_HERBIVORE_GENES, "mutation_rate": hi}
    g2 = {**config.DEFAULT_HERBIVORE_GENES, "mutation_rate": hi}
    child = combine_genes(g1, g2, HERBIVORE)
    assert hi * 0.85 <= child["mutation_rate"] <= hi


def test_color_mutates_additively_within_bounds():
    random.seed(4)
    genes = {**config.DEFAULT_HERBIVORE_GENES, "color": 0.0}
    amount = config.ADDITIVE_GENE_MUTATION["color"]
    seen = set()
    for _ in range(200):
        c = mutate_genes(genes, HERBIVORE)["color"]
        assert 0.0 <= c <= amount
        seen.add(c > 0)
    assert seen == {True, False}  # moves off 0 although a multiplicative step could not


def test_old_gene_dicts_get_defaults_for_new_genes():
    old = {k: config.DEFAULT_HERBIVORE_GENES[k]
           for k in ("size", "max_speed", "sight_distance", "sight_angle", "max_stamina",
                     "attractiveness")}
    child = combine_genes(old, old, HERBIVORE)
    assert set(child) == set(config.DEFAULT_HERBIVORE_GENES)


def test_genetic_distance_is_symmetric_and_range_scaled():
    from evolution.genetics import compatible, genetic_distance
    g = config.DEFAULT_HERBIVORE_GENES
    bounds = config.GENE_MIN_MAX
    assert genetic_distance(g, g, bounds) == 0 and compatible(g, g, HERBIVORE)
    lo, hi = config.GENE_MIN_MAX["color"]
    far = {**g, "color": hi}
    near = {**g, "color": g["color"] + 0.01}
    assert genetic_distance(g, far, bounds) == genetic_distance(far, g, bounds)
    n = len(config.GENE_MIN_MAX)
    assert genetic_distance(g, far, bounds) == pytest.approx(((hi - g["color"]) / (hi - lo)) / n ** 0.5)
    assert compatible(g, near, HERBIVORE)
    assert not compatible({**g, "color": lo, "size": 5}, {**g, "color": hi, "size": 25},
                          HERBIVORE)


def test_combine_brains_takes_each_neuron_from_one_parent():
    import numpy as np
    np.random.seed(3)
    brain1 = NeuralNetwork([4, 6, 2])
    brain2 = NeuralNetwork([4, 6, 2])
    child = combine_brains(brain1, brain2)
    for w, b, w1, b1, w2, b2 in zip(child.weights, child.biases, brain1.weights, brain1.biases,
                                    brain2.weights, brain2.biases):
        for j in range(w.shape[1]):
            from_1 = (w[:, j] == w1[:, j]).all() and b[0, j] == b1[0, j]
            from_2 = (w[:, j] == w2[:, j]).all() and b[0, j] == b2[0, j]
            assert from_1 or from_2
