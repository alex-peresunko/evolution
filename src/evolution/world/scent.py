"""Scent marks: every creature leaves a decaying trail on a coarse grid, one layer per species,
and smells the marks ahead of it through three antennae."""
import numpy as np

from ..config import (
    SCENT_ANTENNA_ANGLE,
    SCENT_ANTENNA_REACH,
    SCENT_CELL_SIZE,
    SCENT_DECAY,
    SCENT_NORM,
)

# Antenna headings relative to the creature's: the negative-angle side, ahead, the positive.
_ANTENNAE = (-1.0, 0.0, 1.0)


class ScentField:
    def __init__(self, world_bounds, n_species, cell_size=SCENT_CELL_SIZE):
        self.cell_size = cell_size
        self.cols = int(world_bounds[0] // cell_size) + 1
        self.rows = int(world_bounds[1] // cell_size) + 1
        self.marks = np.zeros((n_species, self.rows, self.cols))   # [species, row, col]

    def fade(self):
        self.marks *= SCENT_DECAY

    def leave(self, species, xy, amounts):
        """Add amounts[n] to the cell under xy[n], in the layer of species[n]."""
        if len(amounts):
            col = np.clip((xy[:, 0] // self.cell_size).astype(np.intp), 0, self.cols - 1)
            row = np.clip((xy[:, 1] // self.cell_size).astype(np.intp), 0, self.rows - 1)
            np.add.at(self.marks, (species, row, col), amounts)

    def _sample(self, layer, x, y):
        """Marks at (x[n], y[n]) in the layer of species index layer[n], or summed over the
        layers when `layer` is None; 0 outside the world."""
        col = np.floor(x / self.cell_size).astype(np.intp)
        row = np.floor(y / self.cell_size).astype(np.intp)
        inside = (col >= 0) & (col < self.cols) & (row >= 0) & (row < self.rows)
        col, row = np.where(inside, col, 0), np.where(inside, row, 0)
        if layer is None:
            values = self.marks[:, row, col].sum(axis=0)
        else:
            values = self.marks[layer, row, col]
        return np.where(inside, values, 0.0)

    def sense(self, species, x, y, angle):
        """(N, 4) inputs of creatures of species[n] at (x[n], y[n]) heading angle[n]: strength
        and bearing of the own species' marks, then of every other species'. Strength is 0..1
        (tanh of the strongest antenna / SCENT_NORM); bearing is -1..1, positive when the
        marks are stronger on the positive-angle side."""
        own = np.empty((3, len(x)))
        other = np.empty((3, len(x)))
        for k, side in enumerate(_ANTENNAE):
            heading = angle + side * SCENT_ANTENNA_ANGLE
            ax = x + np.cos(heading) * SCENT_ANTENNA_REACH
            ay = y + np.sin(heading) * SCENT_ANTENNA_REACH
            own[k] = self._sample(species, ax, ay)
            other[k] = self._sample(None, ax, ay) - own[k]
        out = np.zeros((len(x), 4))
        for at, antennae in ((0, own), (2, other)):
            total = antennae.sum(axis=0)
            out[:, at] = np.tanh(antennae.max(axis=0) / SCENT_NORM)
            np.divide(antennae[2] - antennae[0], total, out=out[:, at + 1], where=total > 0)
        return out
