from evolution import config
from evolution.species import CARNIVORE, HERBIVORE
from evolution.ui.camera import follow_target, min_zoom, viewport_origin, zoom_at
from evolution.world.creature import Creature
from evolution.world.world import World

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _world():
    herbs = [Creature(HERBIVORE, WORLD_BOUNDS) for _ in range(5)]
    carns = [Creature(CARNIVORE, WORLD_BOUNDS) for _ in range(5)]
    for i, c in enumerate(herbs + carns):
        c.score = float(i)
    return World(WORLD_BOUNDS, {HERBIVORE: herbs, CARNIVORE: carns}, [])


def test_follows_living_creature():
    world = _world()
    c = world.herbivores[1]
    assert follow_target(world, c) == (c, False)


def test_dead_creature_switches_to_best_of_its_species():
    world = _world()
    followed = world.herbivores[1]
    followed.health = 0
    assert follow_target(world, followed)[0] is world.herbivores[-1]  # not the higher-scoring carnivores


def test_removed_creature_switches_to_best():
    world = _world()
    followed = world.carnivores.pop(0)  # replaced by an immigrant, still alive
    assert follow_target(world, followed)[0] is world.carnivores[-1]


def test_nothing_to_follow():
    world = _world()
    followed = world.herbivores[0]
    world.populations[HERBIVORE] = []
    assert follow_target(world, followed) == (None, False)
    assert follow_target(world, None) == (None, False)


def test_pick_from_a_tied_species_is_tentative_until_one_leads():
    world = _world()
    followed = world.herbivores[0]
    for c in world.herbivores:
        c.score = 0.0  # a freshly repopulated species
    world.herbivores.remove(followed)
    pick, tentative = follow_target(world, followed)
    assert tentative
    world.herbivores[2].score = 0.5
    assert follow_target(world, pick, tentative) == (world.herbivores[2], False)
    world.herbivores[3].score = 1.0  # a settled pick sticks, as after a death
    assert follow_target(world, world.herbivores[2]) == (world.herbivores[2], False)


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
