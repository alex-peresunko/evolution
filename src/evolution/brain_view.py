"""Brain introspection for the brain panel in game.py: per-layer activations, edge
contributions and input attributions of one creature's network (pure numpy, no pygame)."""
import numpy as np

from .config import BRAIN_VIEW_EDGE_FRACTION, NUM_MEMORY, NUM_WHISKERS

# Same order as perception.perceive_all fills the input matrix.
INPUT_LABELS = (
    [f"obst_w{i}" for i in range(NUM_WHISKERS)]
    + ["target_ang", "target_dist", "smell_ang", "smell_str", "prey_vx", "prey_vy",
       "health", "sin_ang", "cos_ang", "size", "speed", "sight", "sight_ang", "stamina",
       "hunger", "kin_ang", "kin_dist", "kin_density", "signal_ang", "signal_str",
       "carcass_ang", "carcass_dist", "light"]
    + [f"mem{i}_in" for i in range(NUM_MEMORY)]
    + [f"food_w{i}" for i in range(NUM_WHISKERS)]
    + ["season"]
)
OUTPUT_LABELS = ["turn", "speed", "signal"] + [f"mem{i}" for i in range(NUM_MEMORY)]
NODE_RADIUS_MIN, NODE_RADIUS_MAX = 2, 6


def labels(names, n, prefix):
    """n labels: the known names, padded with prefix+index if the brain has more nodes."""
    return list(names[:n]) + [f"{prefix}{i}" for i in range(len(names), n)]


def network_layout(layer_sizes, left, top, right, bottom):
    """Node centers per layer, with columns spread evenly from left to right and each column's
    nodes evenly from top to bottom, plus a per-layer node radius that fits the spacing."""
    n = len(layer_sizes)
    nodes, radii = [], []
    for k, size in enumerate(layer_sizes):
        x = left + (right - left) * k / (n - 1) if n > 1 else left
        gap = (bottom - top) / size
        nodes.append([(round(x), round(top + (i + 0.5) * gap)) for i in range(size)])
        radii.append(max(NODE_RADIUS_MIN, min(NODE_RADIUS_MAX, int(gap * 0.4))))
    return nodes, radii


def activations(brain, inputs):
    """[inputs, hidden..., outputs] as 1-D arrays; same math as NeuralNetwork.forward."""
    acts = [np.asarray(inputs, dtype=np.float64)]
    for w, b in zip(brain.weights, brain.biases):
        acts.append(np.tanh(acts[-1] @ w + b[0]))
    return acts


def edge_contributions(brain, acts):
    """Per layer, the (n_in, n_out) matrix of weight * source activation."""
    return [w * a[:, None] for w, a in zip(brain.weights, acts)]


def strong_edges(contribs, fraction=BRAIN_VIEW_EDGE_FRACTION):
    """(layer, i, j, value) for the strongest `fraction` of each layer's edges by |value|
    (per layer, so the wide input layer doesn't crowd out the deeper ones)."""
    edges = []
    for layer, c in enumerate(contribs):
        flat = np.abs(c).ravel()
        for idx in np.argsort(-flat)[:max(1, round(len(flat) * fraction))]:
            i, j = np.unravel_index(idx, c.shape)
            edges.append((layer, int(i), int(j), float(c[i, j])))
    return edges


def input_attribution(brain, acts, out_idx):
    """Gradient * input of output out_idx with respect to every input (backprop through tanh)."""
    grad = np.zeros(len(acts[-1]))
    grad[out_idx] = 1.0
    for w, a in zip(reversed(brain.weights), reversed(acts[1:])):
        grad = w @ (grad * (1.0 - a * a))
    return grad * acts[0]


def top_drivers(brain, acts, out_idx, n=3):
    """The n inputs with the largest |attribution| to output out_idx, as (label, value)."""
    attr = input_attribution(brain, acts, out_idx)
    names = labels(INPUT_LABELS, len(attr), "in")
    return [(names[i], float(attr[i])) for i in np.argsort(-np.abs(attr))[:n]]
