"""Building a ready-to-run Simulation, the same way for every entry script."""
from ..config import NUM_FOOD, WORLD_HEIGHT, WORLD_WIDTH
from ..persistence.brain_history import archive_after_start
from ..persistence.brain_store import (
    SAVE_DIR,
    load_brain,
    read_saved_history,
    read_saved_score,
)
from ..sim.populations import founders
from ..sim.records import Record
from ..sim.simulation import Simulation
from ..species import ALL_SPECIES
from ..world.food import Food
from ..world.geometry import HEADLESS
from ..world.world import World
from ..world.worldgen import generate_lakes, generate_obstacles

WORLD_BOUNDS = (WORLD_WIDTH, WORLD_HEIGHT)


def read_progenitor(species, source):
    """The species' saved best brain in folder `source` as a Record (score unset), or None when
    source is None or holds no brain for it."""
    if source is None:
        return None
    try:
        brain, generation, genes = load_brain(species.brain_file, save_dir=source)
    except FileNotFoundError:
        return None
    return Record(score=None, brain=brain, genes=genes, generation=generation)


def load_progenitor(species, source):
    """read_progenitor, announcing which population it starts."""
    progenitor = read_progenitor(species, source)
    if progenitor is None:
        print(f"No saved brain found for {species.plural}. Creating a new random population.")
    else:
        print(f"Loading initial {species.name} population from {species.brain_file} "
              f"(Gen {progenitor.generation}).")
    return progenitor


def start_populations(source, world_bounds=WORLD_BOUNDS, *, geometry=HEADLESS):
    """({species: population}, generation) from the brain files in `source`, or random
    populations when it is None. Each loaded population starts with the unmutated progenitor."""
    populations, generations = {}, []
    for species in ALL_SPECIES:
        progenitor = load_progenitor(species, source)
        populations[species] = founders(species, world_bounds, progenitor=progenitor,
                                        geometry=geometry)
        generations.append(progenitor.generation if progenitor else 1)
    return populations, max(generations)


def build_world(populations, world_bounds=WORLD_BOUNDS, *, geometry=HEADLESS):
    food = [Food(world_bounds, geometry=geometry) for _ in range(NUM_FOOD)]
    obstacles = generate_obstacles(world_bounds, geometry)
    return World(world_bounds, populations, food, obstacles,
                 generate_lakes(world_bounds, obstacles), geometry=geometry)


def restore_progress(sim, species, progenitor, source=SAVE_DIR):
    """Continue where a saved run left off: seed the Hall of Fame with the file's score (so new
    chart samples don't drop to 0) and restore its chart histories. `progenitor` (a Creature or
    Record) is the brain loaded from that file."""
    filename = species.brain_file
    score = read_saved_score(filename, save_dir=source)
    if score is not None:
        sim.hall_of_fame[species].seed(Record(score, progenitor.brain.copy(),
                                              dict(progenitor.genes or {}),
                                              progenitor.generation))
    sim.stats.restore(species, read_saved_history(filename, save_dir=source) or {})


def start_simulation(source, *, geometry=HEADLESS, listeners=(), world_bounds=WORLD_BOUNDS):
    """A Simulation started from the brain files in `source` (None: random brains)."""
    populations, generation = start_populations(source, world_bounds, geometry=geometry)
    sim = Simulation(build_world(populations, world_bounds, geometry=geometry), generation,
                     listeners)
    if source is not None:
        for species in ALL_SPECIES:
            population = sim.world.population(species)
            # population[0] is the unmutated progenitor founders() puts first.
            if population and read_saved_score(species.brain_file, save_dir=source) is not None:
                restore_progress(sim, species, population[0], source)
    return sim


def reload_population(sim, species, source=SAVE_DIR):
    """Replace the species with founders from its brain file in `source` and continue that
    file's progress. Returns the progenitor Record, or None (nothing changed) without a file."""
    progenitor = read_progenitor(species, source)
    if progenitor is not None:
        world = sim.world
        world.populations[species] = founders(species, world.bounds, progenitor=progenitor,
                                              geometry=world.geometry)
        restore_progress(sim, species, progenitor, source)
    return progenitor


def archive_previous_best(source, new, save_dir=SAVE_DIR):
    snapshot = archive_after_start(source, new, save_dir)
    if snapshot:
        print(f"Previous best brains kept in {snapshot}")
