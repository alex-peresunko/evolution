import math


class SpatialHashGrid:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = int(math.ceil(width / cell_size))
        self.rows = int(math.ceil(height / cell_size))
        self.grid = {}

    def clear(self):
        self.grid = {}

    def insert(self, obj):
        # Clamped so entities slightly outside the world (e.g. after an obstacle bounce)
        # stay reachable by the clamped query below.
        key = (min(self.cols - 1, max(0, int(obj.pos.x / self.cell_size))),
               min(self.rows - 1, max(0, int(obj.pos.y / self.cell_size))))
        self.grid.setdefault(key, []).append(obj)

    def query(self, pos, radius):
        min_x = max(0, int((pos.x - radius) / self.cell_size))
        max_x = min(self.cols - 1, int((pos.x + radius) / self.cell_size))
        min_y = max(0, int((pos.y - radius) / self.cell_size))
        max_y = min(self.rows - 1, int((pos.y + radius) / self.cell_size))
        grid = self.grid
        results = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                cell = grid.get((x, y))
                if cell:
                    results.extend(cell)
        return results
