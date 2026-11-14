import numpy as np


class BatchBrain:
    """Runs every creature's brain in one vectorized pass per layer.

    The stacked weights are cached until the population changes, which relies on a creature's
    brain never changing in place after the creature is created."""

    def __init__(self):
        self._key = None
        self._slot = {}
        self._layers = None

    def _stacked_layers(self, creatures):
        key = [(c.id, id(c.brain)) for c in creatures]
        if self._key != key:
            # Rows of creatures stacked last time are copied over; only newcomers are stacked.
            old = [self._slot.get(k, -1) for k in key]
            new = [n for n, o in enumerate(old) if o < 0]
            n_layers = len(creatures[0].brain.weights)
            if self._layers is None or len(new) == len(creatures):
                layers = [
                    (np.stack([c.brain.weights[i] for c in creatures]),
                     np.stack([c.brain.biases[i] for c in creatures])[:, 0, :])
                    for i in range(n_layers)
                ]
            else:
                kept = np.array([o for o in old if o >= 0], dtype=np.intp)
                at = np.array([n for n, o in enumerate(old) if o >= 0], dtype=np.intp)
                layers = []
                for i, (W, b) in enumerate(self._layers):
                    W2 = np.empty((len(creatures),) + W.shape[1:], dtype=W.dtype)
                    b2 = np.empty((len(creatures),) + b.shape[1:], dtype=b.dtype)
                    W2[at], b2[at] = W[kept], b[kept]
                    if new:
                        W2[new] = np.stack([creatures[n].brain.weights[i] for n in new])
                        b2[new] = np.stack([creatures[n].brain.biases[i] for n in new])[:, 0, :]
                    layers.append((W2, b2))
            self._layers = layers
            self._key = key
            self._slot = {k: n for n, k in enumerate(key)}
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
