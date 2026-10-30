import os
import tempfile

import numpy as np
import pytest

from evolution.neural_network import NeuralNetwork


def test_forward_output_shape():
    nn = NeuralNetwork([4, 3, 2])
    out = nn.forward([0.0] * 4)
    assert len(out) == 2


def test_forward_values_bounded():
    nn = NeuralNetwork([4, 3, 2])
    out = nn.forward([0.5, -0.5, 1.0, -1.0])
    assert all(-1.0 < v < 1.0 for v in out)


def test_mutate_changes_weights():
    np.random.seed(42)
    nn = NeuralNetwork([4, 3, 2])
    original = [w.copy() for w in nn.weights]
    nn.mutate(1.0, 0.5)
    assert any(not np.allclose(o, n) for o, n in zip(original, nn.weights))


def test_copy_is_independent():
    np.random.seed(0)
    nn = NeuralNetwork([4, 3, 2])
    copy = nn.copy()
    copy.mutate(1.0, 1.0)
    assert not np.allclose(nn.weights[0], copy.weights[0])


def test_save_load_roundtrip():
    np.random.seed(7)
    nn = NeuralNetwork([4, 3, 2])
    inputs = [0.1, 0.2, -0.3, 0.4]
    expected = nn.forward(inputs)
    with tempfile.TemporaryDirectory() as tmpdir:
        nn.save("test_brain.json", save_dir=tmpdir, generation=5, score=3.14)
        nn2, gen, genes = NeuralNetwork.load("test_brain.json", [4, 3, 2], save_dir=tmpdir)
    assert gen == 5
    assert np.allclose(nn2.forward(inputs), expected)


def test_load_topology_mismatch_no_crash():
    np.random.seed(3)
    nn = NeuralNetwork([4, 3, 2])
    with tempfile.TemporaryDirectory() as tmpdir:
        nn.save("brain.json", save_dir=tmpdir)
        nn2, _, _ = NeuralNetwork.load("brain.json", [6, 3, 2], save_dir=tmpdir)
    assert nn2.layer_sizes == [6, 3, 2]
    out = nn2.forward([0.0] * 6)
    assert len(out) == 2


def test_random_brains_behave_differently():
    np.random.seed(11)
    x = np.random.uniform(-1, 1, 32)
    turns = [NeuralNetwork([32, 16, 8, 5]).forward(x)[0] for _ in range(50)]
    assert np.std(turns) > 0.2
