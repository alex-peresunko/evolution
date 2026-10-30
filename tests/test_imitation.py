import math

import numpy as np
import pytest

from evolution import config
from evolution.imitation import (
    CARCASS_ANGLE,
    CARCASS_DIST,
    SIGHT_ANGLE,
    SMELL_ANGLE,
    SMELL_STRENGTH,
    STAMINA,
    TARGET_ANGLE,
    TARGET_DIST,
    _gradients,
    loss,
    teacher_actions,
    train_brain,
)
from evolution.neural_network import NeuralNetwork


def _row():
    x = np.zeros(config.BRAIN_INPUTS)
    x[TARGET_DIST] = x[CARCASS_DIST] = 1.0  # nothing seen
    x[SIGHT_ANGLE] = 0.5                    # 180 degree cone
    x[STAMINA] = 1.0
    return x


def _act(row, carnivore=False):
    return teacher_actions(row[None, :], [carnivore])[0]


@pytest.mark.parametrize("side", [-1, 1])
def test_turns_toward_visible_target(side):
    row = _row()
    row[TARGET_DIST], row[TARGET_ANGLE] = 0.3, 0.2 * side
    assert np.sign(_act(row)[0]) == side


def test_herbivore_flees_predator_smell_behind():
    row = _row()
    row[SMELL_STRENGTH], row[SMELL_ANGLE] = 0.9, 0.9  # predator behind-left (positive)
    rotation, speed = _act(row)[:2]
    assert rotation < 0 and speed > 0.9


def test_carnivore_follows_smell_then_carcass():
    row = _row()
    row[SMELL_STRENGTH], row[SMELL_ANGLE] = 0.5, -0.1
    assert _act(row, carnivore=True)[0] < 0
    row[CARCASS_DIST], row[CARCASS_ANGLE] = 0.5, 0.4
    assert _act(row, carnivore=True)[0] > 0


def test_near_obstacle_overrides_target():
    row = _row()
    row[TARGET_DIST], row[TARGET_ANGLE] = 0.3, -0.5
    row[0] = 0.9  # left whisker almost touching
    assert _act(row)[0] > 0.9


def test_signal_and_memory_targets_are_zero():
    rng = np.random.default_rng(0)
    Y = teacher_actions(rng.uniform(-1, 1, (20, config.BRAIN_INPUTS)), [False] * 20)
    assert Y.shape == (20, config.BRAIN_OUTPUTS)
    assert (Y[:, 2:] == 0).all() and (np.abs(Y) < 1).all()


def test_gradients_match_finite_differences():
    np.random.seed(0)
    brain = NeuralNetwork([4, 3, 2])
    X, Y = np.random.uniform(-1, 1, (8, 4)), np.random.uniform(-0.9, 0.9, (8, 2))
    grads = _gradients(brain, X, Y)
    params = [p for pair in zip(brain.weights, brain.biases) for p in pair]
    eps = 1e-6
    for p, g in zip(params, grads):
        p[0, 0] += eps
        up = loss(brain, X, Y)
        p[0, 0] -= 2 * eps
        down = loss(brain, X, Y)
        p[0, 0] += eps
        assert g[0, 0] == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-9)


def test_trained_brain_imitates_teacher():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (2000, config.BRAIN_INPUTS))
    X[:, TARGET_DIST] = rng.uniform(0, 0.9, len(X))
    X[:, SMELL_STRENGTH] = 0.0
    X[:, :config.NUM_WHISKERS] = 0.0
    X[:, SIGHT_ANGLE] = 0.5
    Y = teacher_actions(X, np.zeros(len(X), bool))
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    before = loss(brain, X, Y)
    after = train_brain(brain, X, Y, steps=1500, rng=np.random.default_rng(2))
    assert after < before / 5
    left, right = X[:1].copy(), X[:1].copy()
    left[0, TARGET_ANGLE], right[0, TARGET_ANGLE] = -0.6, 0.6
    assert brain.forward(left[0])[0] < 0 < brain.forward(right[0])[0]
    assert math.isfinite(after)
