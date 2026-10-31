import numpy as np


class BatchBrain:
    """Runs every creature's brain in one vectorized pass per layer.

    The stacked weights are cached until the population changes, which relies on a creature's
    brain never changing in place after the creature is created."""

    def __init__(self):
        self._key = None
        self._layers = None

    def _stacked_layers(self, creatures):
        key = tuple((c.id, id(c.brain)) for c in creatures)
        if self._key != key:
            n_layers = len(creatures[0].brain.weights)
            self._layers = [
                (np.stack([c.brain.weights[i] for c in creatures]),
                 np.stack([c.brain.biases[i] for c in creatures])[:, 0, :])
                for i in range(n_layers)
            ]
            self._key = key
        return self._layers

    def forward(self, creatures, inputs):
        """(N, outputs) brain outputs for the (N, inputs) rows of `creatures`."""
        if not creatures:
            return []
        X = np.asarray(inputs, dtype=np.float64)
        layers = self._stacked_layers(creatures)
        n_layers = len(layers)
        for layer_idx, (W, b) in enumerate(layers):
            X = np.einsum("ni,nij->nj", X, W) + b
            if layer_idx < n_layers - 1:
                X = np.tanh(X)
        return np.tanh(X)
