import math
import operator

import numpy as np
from numba import njit

_is = operator.is_
_brain = operator.attrgetter("brain")


@njit(cache=True)
def _forward(X, params, slots, sizes, out):
    """out[n] = the brain in params[slots[n]] applied to X[n]. A params row holds, layer after
    layer, the (inputs, outputs) weights row by row and then the biases; `sizes` are the layer
    widths."""
    widest = sizes.max()
    a = np.empty(widest)
    h = np.empty(widest)
    layers = sizes.shape[0] - 1
    for n in range(X.shape[0]):
        p = params[slots[n]]
        a[:sizes[0]] = X[n]
        at = 0
        for layer in range(layers):
            n_in, n_out = sizes[layer], sizes[layer + 1]
            h[:n_out] = 0.0
            for i in range(n_in):
                x = a[i]
                if x != 0.0:
                    row = p[at + i * n_out:at + (i + 1) * n_out]
                    for j in range(n_out):
                        h[j] += x * row[j]
            at += n_in * n_out
            for j in range(n_out):
                h[j] = math.tanh(h[j] + p[at + j])
            at += n_out
            a, h = h, a
        out[n] = a[:sizes[layers]]


class BatchBrain:
    """Runs every creature's brain in one compiled pass.

    Each brain's weights are copied once into a slot of a shared array and stay there while a
    creature in the population has that brain, which relies on a brain never changing in place
    after its creature is created."""

    def __init__(self):
        self._creatures = ()
        self._brains = ()
        self._sizes = None
        self._reset(0)

    def _reset(self, width):
        self._slot = {}      # id(brain) -> (slot, brain); the reference keeps the id unique
        self._free = []
        self._params = np.empty((0, width))
        self._slots = np.empty(0, dtype=np.intp)

    def _sync(self, creatures):
        if (len(creatures) == len(self._creatures)
                and all(map(_is, creatures, self._creatures))
                and all(map(_is, map(_brain, creatures), self._brains))):
            return
        brains = [c.brain for c in creatures]
        first = brains[0].weights
        sizes = [w.shape[0] for w in first] + [first[-1].shape[1]]
        if sizes != self._sizes:
            self._sizes = sizes
            self._reset(sum(w.size + w.shape[1] for w in first))
        slot_of, free = self._slot, self._free
        present = set(map(id, brains))
        for key in [key for key in slot_of if key not in present]:
            free.append(slot_of.pop(key)[0])
        slots = []
        for brain in brains:
            entry = slot_of.get(id(brain))
            if entry is None:
                if not free:
                    old = self._params
                    self._params = np.empty((max(64, 2 * len(old)), old.shape[1]))
                    self._params[:len(old)] = old
                    free.extend(range(len(self._params) - 1, len(old) - 1, -1))
                slot = free.pop()
                self._params[slot] = np.concatenate(
                    [part.ravel() for layer in zip(brain.weights, brain.biases) for part in layer])
                entry = slot_of[id(brain)] = (slot, brain)
            slots.append(entry[0])
        self._slots = np.array(slots, dtype=np.intp)
        self._creatures, self._brains = list(creatures), brains

    def forward(self, creatures, inputs):
        """(N, outputs) brain outputs for the (N, inputs) rows of `creatures`."""
        if not creatures:
            return []
        self._sync(creatures)
        sizes = np.array(self._sizes, dtype=np.intp)
        out = np.empty((len(creatures), sizes[-1]))
        _forward(np.ascontiguousarray(inputs, dtype=np.float64), self._params, self._slots,
                 sizes, out)
        return out
