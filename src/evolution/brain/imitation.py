"""Behavior cloning: a hand-written teacher policy labels perception rows, and a brain is fit to
those labels with backprop, so evolution starts from creatures that already seek food and flee."""
import math

import numpy as np

from ..config import CREATURE_ROTATION_SPEED
from .layout import INPUTS as IN
from .layout import OUTPUTS as OUT

OBSTACLE_NEAR = 0.7      # whisker proximity that overrides steering
FLEE_SMELL = 0.6         # predator smell strength that makes a herbivore run
CRUISE_SPEED = 0.55      # below STAMINA_EFFORT_THRESHOLD, so cruising is free
SPRINT_MIN_STAMINA = 0.2
WANDER_TURN = 0.05
TARGET_LIMIT = 0.95      # tanh outputs never reach +-1
CROWDED = 0.5            # kin touch strength (bodies in contact) that makes a wanderer step aside


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def teacher_actions(X, is_carnivore):
    """(N, OUTPUTS.width) target outputs for perception rows X; signal and memory targets are 0.
    A fleeing herbivore abandons its meal; nothing else does."""
    X = np.asarray(X, dtype=np.float64)
    carn = np.asarray(is_carnivore, dtype=bool)
    half = X[:, IN.sight_angle] * math.pi

    seen = X[:, IN.target_dist] < 1
    rel = np.where(seen, X[:, IN.target_angle] * half, WANDER_TURN * CREATURE_ROTATION_SPEED)

    carcass = carn & ~seen & (X[:, IN.carcass_dist] < 1)
    rel = np.where(carcass, X[:, IN.carcass_angle] * half, rel)
    tracking = carn & ~seen & ~carcass & (X[:, IN.smell_strength] > 0)
    rel = np.where(tracking, X[:, IN.smell_angle] * math.pi, rel)
    carrion = carn & ~seen & ~carcass & ~tracking & (X[:, IN.carcass_smell_strength] > 0)
    rel = np.where(carrion, X[:, IN.carcass_smell_angle] * math.pi, rel)

    wandering = ~seen & ~carcass & ~tracking & ~carrion
    crowded = wandering & (X[:, IN.touch_kin] >= CROWDED)
    rel = np.where(crowded, _wrap(X[:, IN.touch_kin_angle] * math.pi + math.pi), rel)

    touched = ~carn & (X[:, IN.touch_other] > 0)
    fleeing = ~carn & ((X[:, IN.smell_strength] > FLEE_SMELL) | (X[:, IN.hurt] > 0) | touched)
    rel = np.where(fleeing, _wrap(X[:, IN.smell_angle] * math.pi + math.pi), rel)
    rel = np.where(touched, _wrap(X[:, IN.touch_other_angle] * math.pi + math.pi), rel)

    rotation = np.clip(rel / CREATURE_ROTATION_SPEED, -1, 1)
    whiskers = X[:, IN.obstacle_whiskers]
    left, centre, right = whiskers[:, 0], whiskers[:, 1], whiskers[:, -1]
    blocked = np.maximum.reduce([left, centre, right]) > OBSTACLE_NEAR
    # Whisker 0 points to the negative-angle side, so a near left obstacle means turn positive.
    away = np.where(np.abs(left - right) > 0.05, np.sign(left - right), 1.0)
    rotation = np.where(blocked, away, rotation)

    sprint = (fleeing | (carn & seen)) & (X[:, IN.stamina] > SPRINT_MIN_STAMINA)
    speed = np.where(sprint, 1.0, CRUISE_SPEED)

    Y = np.zeros((len(X), OUT.width))
    Y[:, OUT.rotation], Y[:, OUT.speed] = rotation, speed
    Y[:, OUT.stop_eating] = np.where(fleeing, 1.0, -1.0)
    return np.clip(Y, -TARGET_LIMIT, TARGET_LIMIT)


def train_brain(brain, X, Y, steps=4000, batch_size=256, lr=3e-3, rng=None):
    """Fit brain (in place) to Y = brain.forward(X) by Adam on mean squared error.
    Returns the final full-data loss."""
    rng = rng or np.random.default_rng()
    params = [p for pair in zip(brain.weights, brain.biases) for p in pair]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(X), batch_size)
        grads = _gradients(brain, X[idx], Y[idx])
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            m_hat = m[i] / (1 - beta1 ** step)
            v_hat = v[i] / (1 - beta2 ** step)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return loss(brain, X, Y)


def _activations(brain, X):
    hs = [X]
    for W, b in zip(brain.weights, brain.biases):
        hs.append(np.tanh(hs[-1] @ W + b))
    return hs


def loss(brain, X, Y):
    return float(np.mean((_activations(brain, X)[-1] - Y) ** 2))


def _gradients(brain, X, Y):
    """Gradients of the MSE, ordered like train_brain's params: (dW0, db0, dW1, db1, ...)."""
    hs = _activations(brain, X)
    dh = 2 * (hs[-1] - Y) / Y.size
    grads = []
    for i in reversed(range(len(brain.weights))):
        dz = dh * (1 - hs[i + 1] ** 2)
        grads.append(dz.sum(axis=0, keepdims=True))
        grads.append(hs[i].T @ dz)
        dh = dz @ brain.weights[i].T
    return grads[::-1]
