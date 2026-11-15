import math

import numpy as np


class SpatialHashGrid:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = int(math.ceil(width / cell_size))
        self.rows = int(math.ceil(height / cell_size))
        self.grid = {}
        self._queries = {}

    def clear(self):
        self.grid = {}
        self._queries = {}

    def insert(self, obj):
        self.grid.setdefault(self.cell_of(obj.pos), []).append(obj)
        self._queries = {}

    def remove(self, obj):
        """Take out an object inserted earlier; it must not have moved since."""
        self.grid[self.cell_of(obj.pos)].remove(obj)
        self._queries = {}

    def cell_of(self, pos):
        # Clamped so entities slightly outside the world (e.g. after an obstacle bounce)
        # stay reachable by the clamped query below.
        return (min(self.cols - 1, max(0, int(pos.x / self.cell_size))),
                min(self.rows - 1, max(0, int(pos.y / self.cell_size))))

    def cell_range(self, pos, radius):
        """(min_x, max_x, min_y, max_y) of the cells the square around pos touches."""
        return (max(0, int((pos.x - radius) / self.cell_size)),
                min(self.cols - 1, int((pos.x + radius) / self.cell_size)),
                max(0, int((pos.y - radius) / self.cell_size)),
                min(self.rows - 1, int((pos.y + radius) / self.cell_size)))

    def query(self, pos, radius):
        """Everything in the cells the square around pos touches, as a tuple; the same tuple
        until the next insert or clear. Ordered by cell (x, then y), then by insertion."""
        key = self.cell_range(pos, radius)
        results = self._queries.get(key)
        if results is None:
            min_x, max_x, min_y, max_y = key
            grid = self.grid
            found = []
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    cell = grid.get((x, y))
                    if cell:
                        found.extend(cell)
            results = self._queries[key] = tuple(found)
        return results


class SpatialIndex:
    """Spatial grids of the food, each species and the carcasses, as of the last rebuild().

    The food grid is kept between rebuilds, which relies on food never moving. A rebuild only
    adds the items appended to the food list since the last one and takes out those reported
    through food_removed(); after any other change to the list it starts over.

    The species and carcass grids are filled on first use after a rebuild, from the members
    and positions the rebuild saw."""

    def __init__(self, world_bounds, species, cell_size):
        w, h = world_bounds
        self.food = SpatialHashGrid(w, h, cell_size)
        self._carcasses = SpatialHashGrid(w, h, cell_size)
        self._species = {s: SpatialHashGrid(w, h, cell_size) for s in species}
        self._unfilled = {}      # grid id -> (grid, items, positions or None) not yet filled
        self._food_xy = np.zeros((0, 2))
        self._food_list = None   # the list last indexed, while its changes are all known
        self._food_last = None   # its last indexed item
        self._food_gone = None   # (rows of _food_xy, items) removed since the last rebuild

    def of(self, species):
        return self._filled(self._species[species])

    @property
    def carcasses(self):
        return self._filled(self._carcasses)

    def _filled(self, grid):
        if self._unfilled:
            pending = self._unfilled.pop(id(grid), None)
            if pending is not None:
                _, items, xy = pending
                if xy is None:
                    _fill(grid, items)
                else:
                    _fill_at(grid, items, xy)
        return grid

    def rebuild(self, food, populations, carcasses, positions=None):
        """`positions`: optional {species: (N, 2) array} of the populations' positions, which
        the caller must not change afterwards."""
        self._sync_food(food)
        for species, grid in self._species.items():
            if positions is None:
                _fill(grid, populations[species])
            else:
                self._unfilled[id(grid)] = (grid, list(populations[species]), positions[species])
        self._unfilled[id(self._carcasses)] = (self._carcasses, list(carcasses), None)

    def food_xy(self, food):
        """(F, 2) positions of the items of `food`, or None unless `food` is still the list as
        the last rebuild() saw it."""
        n = len(self._food_xy)
        if (food is self._food_list and self._food_gone is None and len(food) == n
                and (n == 0 or food[-1] is self._food_last)):
            return self._food_xy
        return None

    def food_removed(self, food, indices, items):
        """Report that `items`, which were at the ascending `indices` of `food`, were just
        deleted from it. The grid keeps them until the next rebuild()."""
        n = len(self._food_xy)
        if (food is not self._food_list or self._food_gone is not None
                or (indices and indices[-1] >= n)):
            self._food_list = None
            return
        self._food_gone = (indices, items)
        n -= len(indices)
        self._food_last = food[n - 1] if n else None

    def _sync_food(self, food):
        gone, self._food_gone = self._food_gone, None
        n = len(self._food_xy) - (len(gone[0]) if gone else 0)
        if (food is not self._food_list or len(food) < n
                or (n and food[n - 1] is not self._food_last)):
            _fill(self.food, food)
            self._food_xy = _xy(food)
        else:
            if gone:
                for item in gone[1]:
                    self.food.remove(item)
                self._food_xy = np.delete(self._food_xy, gone[0], axis=0)
            if len(food) > n:
                new = food[n:]
                for item in new:
                    self.food.insert(item)
                self._food_xy = np.concatenate((self._food_xy, _xy(new)))
        self._food_list = food
        self._food_last = food[-1] if food else None


def _xy(items):
    return np.array([(o.pos.x, o.pos.y) for o in items], dtype=np.float64).reshape(-1, 2)


def _fill(grid, items):
    grid.clear()
    for item in items:
        grid.insert(item)


def _fill_at(grid, items, xy):
    """_fill, with the items' positions given as an (N, 2) array: the same cells, computed as
    cell_of does (int() truncates toward zero, like astype)."""
    grid.clear()
    if not len(items):
        return
    cx = np.minimum(grid.cols - 1, np.maximum(0, (xy[:, 0] / grid.cell_size).astype(np.int64)))
    cy = np.minimum(grid.rows - 1, np.maximum(0, (xy[:, 1] / grid.cell_size).astype(np.int64)))
    cells = grid.grid
    for item, cell in zip(items, zip(cx.tolist(), cy.tolist())):
        bucket = cells.get(cell)
        if bucket is None:
            cells[cell] = [item]
        else:
            bucket.append(item)
