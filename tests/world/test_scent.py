import math

import numpy as np
import pytest

from evolution import config
from evolution.brain.layout import INPUTS
from evolution.species import CARNIVORE, HERBIVORE
from evolution.world.creature import Creature
from evolution.world.scent import ScentField
from evolution.world.world import World

BOUNDS = (800, 600)
HERB, CARN = 0, 1    # layers, in ALL_SPECIES order


def _sense(field, species, x, y, angle=0.0):
    return field.sense(np.array([species]), np.array([float(x)]), np.array([float(y)]),
                       np.array([angle]))[0]


def test_marks_land_in_the_species_layer_and_fade():
    field = ScentField(BOUNDS, 2)
    field.leave(np.array([HERB, CARN, CARN]), np.array([[10.0, 10.0], [410.0, 90.0], [415.0, 95.0]]),
                np.array([1.0, 2.0, 3.0]))
    cell = config.SCENT_CELL_SIZE
    assert field.marks[HERB, 0, 0] == 1.0
    assert field.marks[CARN, 90 // cell, 410 // cell] == 5.0
    assert field.marks.sum() == 6.0
    field.fade()
    assert field.marks.sum() == pytest.approx(6.0 * config.SCENT_DECAY)


def test_a_mark_ahead_is_smelled_by_both_species_in_their_own_input_pair():
    field = ScentField(BOUNDS, 2)
    ahead = np.array([[400.0 + config.SCENT_ANTENNA_REACH, 300.0]])
    field.leave(np.array([HERB]), ahead, np.array([config.SCENT_NORM]))
    kin, kin_bearing, other, other_bearing = _sense(field, HERB, 400, 300)
    assert kin == pytest.approx(math.tanh(1.0)) and kin_bearing == 0.0
    assert (other, other_bearing) == (0.0, 0.0)
    kin, _, other, _ = _sense(field, CARN, 400, 300)
    assert kin == 0.0 and other == pytest.approx(math.tanh(1.0))


def test_bearing_points_to_the_side_with_the_stronger_marks():
    field = ScentField(BOUNDS, 2)
    a = config.SCENT_ANTENNA_ANGLE
    side = np.array([[400.0 + math.cos(a) * config.SCENT_ANTENNA_REACH,
                      300.0 + math.sin(a) * config.SCENT_ANTENNA_REACH]])
    field.leave(np.array([HERB]), side, np.array([5.0]))
    assert _sense(field, HERB, 400, 300)[1] == 1.0
    assert _sense(field, HERB, 400, 300, angle=2 * a)[1] == -1.0


def test_nothing_is_smelled_beyond_the_world_edge():
    field = ScentField(BOUNDS, 2)
    field.marks[:] = 50.0
    kin, _, other, _ = _sense(field, HERB, BOUNDS[0] - 1, 300)   # every antenna outside
    assert (kin, other) == (0.0, 0.0)


def test_creatures_leave_a_trail_and_mark_on_top_of_it():
    herb, carn = Creature(HERBIVORE, BOUNDS), Creature(CARNIVORE, BOUNDS)
    herb.pos, carn.pos = herb.geometry.vec2(100, 100), carn.geometry.vec2(500, 400)
    world = World(BOUNDS, {HERBIVORE: [herb], CARNIVORE: [carn]}, [])
    world.leave_scent([0.0, 1.0])
    cell = config.SCENT_CELL_SIZE
    assert world.scent.marks[HERB, 100 // cell, 100 // cell] == config.SCENT_TRAIL
    assert world.scent.marks[CARN, 400 // cell, 500 // cell] == (
        config.SCENT_TRAIL + config.SCENT_MARK)
    assert world.scent.marks.sum() == 2 * config.SCENT_TRAIL + config.SCENT_MARK


def test_perception_reads_the_worlds_scent():
    herb = Creature(HERBIVORE, BOUNDS)
    herb.pos, herb.angle = herb.geometry.vec2(400, 300), 0.0
    world = World(BOUNDS, {HERBIVORE: [herb]}, [])
    world.scent.marks[CARN] = 3 * config.SCENT_NORM
    row = world.perceive()[0]
    assert row[INPUTS.trail_other] == pytest.approx(math.tanh(3.0))
    assert row[INPUTS.trail_kin] == 0.0
