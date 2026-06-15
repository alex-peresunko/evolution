import pytest

from evolution.spatial import SpatialHashGrid
from evolution.math_types import Vector2


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
    assert results == []


def test_query_boundary_cell():
    grid = SpatialHashGrid(1000, 1000, 100)
    # Place object exactly on a cell boundary
    obj = _Obj(100, 100)
    grid.insert(obj)
    # Query from slightly to the left — should still find it
    results = grid.query(Vector2(99, 99), 10)
    assert obj in results
