from evolution import config
from evolution.camera import follow_target, min_zoom, viewport_origin, zoom_at
from evolution.entities import Creature
from evolution.simulation import build_sim_state

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _sim():
    herbs = [Creature(WORLD_BOUNDS) for _ in range(5)]
    carns = [Creature(WORLD_BOUNDS, is_carnivore=True) for _ in range(5)]
    for i, c in enumerate(herbs + carns):
        c.score = float(i)
    return build_sim_state(WORLD_BOUNDS, herbs, carns, [], [])


def test_follows_living_creature():
    sim = _sim()
    c = sim["creatures"][1]
    assert follow_target(sim, c) is c


def test_dead_creature_switches_to_best_of_its_species():
    sim = _sim()
    followed = sim["creatures"][1]
    followed.health = 0
    assert follow_target(sim, followed) is sim["creatures"][-1]  # not the higher-scoring carnivores


def test_removed_creature_switches_to_best():
    sim = _sim()
    followed = sim["carnivores"].pop(0)  # replaced by an immigrant, still alive
    assert follow_target(sim, followed) is sim["carnivores"][-1]


def test_nothing_to_follow():
    sim = _sim()
    followed = sim["creatures"][0]
    sim["creatures"] = []
    assert follow_target(sim, followed) is None
    assert follow_target(sim, None) is None


def test_viewport_origin_centers_and_clamps():
    world, screen = (4000, 2000), (1000, 500)
    assert viewport_origin(2000, 1000, world, screen) == (1500, 750)
    assert viewport_origin(10, 10, world, screen) == (0, 0)
    assert viewport_origin(3990, 1990, world, screen) == (3000, 1500)
    assert viewport_origin(500, 300, screen, screen) == (0, 0)


def test_min_zoom_fits_the_world():
    assert min_zoom((4000, 2000), (1000, 500)) == 0.25
    assert min_zoom((1920, 900), (1920, 900)) == 1.0


def test_zoom_keeps_the_point_under_the_cursor():
    world, screen = (4000, 2000), (1000, 500)
    cam = zoom_at((1000, 500), 1.0, 2.0, (300, 200), world, screen)
    assert cam == (1150, 600)  # world (1300, 700) is still under screen (300, 200)
    assert zoom_at(cam, 2.0, 1.0, (300, 200), world, screen) == (1000, 500)


def test_zoom_out_clamps_to_the_world():
    world = screen = (1000, 500)
    assert zoom_at((400, 200), 2.0, 1.0, (500, 250), world, screen) == (0, 0)
