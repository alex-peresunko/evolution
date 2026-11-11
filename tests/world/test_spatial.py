
import random

from evolution.world.geometry import Vector2
from evolution.world.spatial import SpatialHashGrid, SpatialIndex


class _Obj:
    def __init__(self, x, y):
        self.pos = Vector2(x, y)


def test_insert_and_query():
    grid = SpatialHashGrid(1000, 1000, 100)
    a = _Obj(50, 50)
    b = _Obj(150, 150)
    c = _Obj(900, 900)
    for obj in (a, b, c):
        grid.insert(obj)
    results = grid.query(Vector2(100, 100), 120)
    assert a in results
    assert b in results
    assert c not in results


def test_clear():
    grid = SpatialHashGrid(500, 500, 100)
    grid.insert(_Obj(50, 50))
    grid.clear()
    results = grid.query(Vector2(50, 50), 200)
    assert results == ()


def test_query_boundary_cell():
    grid = SpatialHashGrid(1000, 1000, 100)
    # Place object exactly on a cell boundary
    obj = _Obj(100, 100)
    grid.insert(obj)
    # Query from slightly to the left — should still find it
    results = grid.query(Vector2(99, 99), 10)
    assert obj in results


def _cells(grid):
    return {key: cell for key, cell in grid.grid.items() if cell}


def _index_and_fresh(food):
    """An index rebuilt after changes to `food` next to one built from scratch."""
    fresh = SpatialIndex((1000, 1000), [], 100)
    fresh.rebuild(food, {}, [])
    return fresh


def test_food_grid_follows_appends_and_reported_removals():
    rng = random.Random(4)
    food = [_Obj(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(200)]
    index = SpatialIndex((1000, 1000), [], 100)
    index.rebuild(food, {}, [])
    for _ in range(20):
        gone = sorted(rng.sample(range(len(food)), 7))
        items = [food[i] for i in gone]
        for i in reversed(gone):
            del food[i]
        index.food_removed(food, gone, items)
        assert index.food_xy(food) is None       # stale until the rebuild
        assert items[0] in index.food.query(items[0].pos, 1)
        food.extend(_Obj(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(5))
        index.rebuild(food, {}, [])
        fresh = _index_and_fresh(food)
        assert _cells(index.food) == _cells(fresh.food)
        assert index.food_xy(food).tolist() == [[f.pos.x, f.pos.y] for f in food]


def test_food_grid_starts_over_after_an_unreported_change():
    food = [_Obj(10 + i, 10) for i in range(5)]
    index = SpatialIndex((1000, 1000), [], 100)
    index.rebuild(food, {}, [])
    del food[1]
    assert index.food_xy(food) is None
    index.rebuild(food, {}, [])
    assert _cells(index.food) == _cells(_index_and_fresh(food).food)
    replaced = [_Obj(500, 500)]
    index.rebuild(replaced, {}, [])
    assert index.food.query(Vector2(10, 10), 1) == ()
    assert index.food_xy(replaced).tolist() == [[500, 500]]
