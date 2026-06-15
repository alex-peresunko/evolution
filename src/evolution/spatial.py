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
        key = (int(obj.pos.x / self.cell_size), int(obj.pos.y / self.cell_size))
        self.grid.setdefault(key, []).append(obj)

    def query(self, pos, radius):
        min_x = int((pos.x - radius) / self.cell_size)
        max_x = int((pos.x + radius) / self.cell_size)
        min_y = int((pos.y - radius) / self.cell_size)
        max_y = int((pos.y + radius) / self.cell_size)
        results = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                results.extend(self.grid.get((x, y), []))
        return results
