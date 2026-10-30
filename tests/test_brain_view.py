import numpy as np

from evolution import config
from evolution.brain_view import (
    INPUT_LABELS,
    NODE_RADIUS_MAX,
    OUTPUT_LABELS,
    activations,
    edge_contributions,
    input_attribution,
    labels,
    network_layout,
    strong_edges,
    top_drivers,
)
from evolution.entities import Creature, Food
from evolution.neural_network import NeuralNetwork
from evolution.simulation import batch_forward, build_sim_state, update_world

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _brain_and_inputs(seed=0):
    rng = np.random.default_rng(seed)
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    brain.weights = [rng.normal(0, 0.5, w.shape) for w in brain.weights]
    return brain, rng.uniform(-1, 1, config.BRAIN_INPUTS)


def test_labels_match_topology():
    assert len(INPUT_LABELS) == config.BRAIN_INPUTS
    assert len(OUTPUT_LABELS) == config.BRAIN_OUTPUTS


def test_activations_match_forward_and_batch_forward():
    brain, x = _brain_and_inputs()
    acts = activations(brain, x)
    assert [len(a) for a in acts] == config.BRAIN_TOPOLOGY
    np.testing.assert_allclose(acts[-1], brain.forward(x))
    c = Creature(WORLD_BOUNDS, brain=brain)
    np.testing.assert_allclose(acts[-1], batch_forward([c], [x])[0])


def test_attribution_matches_finite_differences():
    brain, x = _brain_and_inputs(1)
    acts = activations(brain, x)
    eps = 1e-6
    for out_idx in range(config.BRAIN_OUTPUTS):
        numeric = np.array([
            (brain.forward(x + eps * e)[out_idx] - brain.forward(x - eps * e)[out_idx]) / (2 * eps)
            for e in np.eye(len(x))
        ]) * x
        np.testing.assert_allclose(input_attribution(brain, acts, out_idx), numeric, atol=1e-6)
    drivers = top_drivers(brain, acts, 0)
    assert len(drivers) == 3 and abs(drivers[0][1]) >= abs(drivers[-1][1])


def test_strong_edges_are_the_largest():
    brain, x = _brain_and_inputs(2)
    contribs = edge_contributions(brain, activations(brain, x))
    edges = strong_edges(contribs, fraction=0.15)
    for k, c in enumerate(contribs):
        kept = [(i, j) for layer, i, j, _ in edges if layer == k]
        assert len(kept) == max(1, round(c.size * 0.15))
        weakest_kept = min(abs(c[i, j]) for i, j in kept)
        for (i, j), v in np.ndenumerate(c):
            if (i, j) not in kept:
                assert abs(v) <= weakest_kept


def test_update_world_records_brain_rows():
    herbs = [Creature(WORLD_BOUNDS) for _ in range(3)]
    carns = [Creature(WORLD_BOUNDS, is_carnivore=True)]
    sim = build_sim_state(WORLD_BOUNDS, herbs, carns, [Food(WORLD_BOUNDS) for _ in range(5)], [])
    assert herbs[0].brain_inputs is None
    update_world(sim)
    for c in sim["creatures"] + sim["carnivores"]:
        assert len(c.brain_inputs) == config.BRAIN_INPUTS
        assert len(c.brain_outputs) == config.BRAIN_OUTPUTS
        np.testing.assert_allclose(c.brain_outputs, c.brain.forward(c.brain_inputs))


def test_layout_follows_topology():
    nodes, radii = network_layout([40, 32, 24, 12, 7], 0, 0, 400, 400)
    assert [len(col) for col in nodes] == [40, 32, 24, 12, 7]
    xs = [col[0][0] for col in nodes]
    assert xs == sorted(xs) and xs[0] == 0 and xs[-1] == 400
    assert all(0 <= y <= 400 for col in nodes for _, y in col)
    assert radii[0] < radii[-1] <= NODE_RADIUS_MAX  # dense layers get smaller nodes
    assert labels(["a", "b"], 4, "in") == ["a", "b", "in2", "in3"]
