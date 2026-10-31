"""Where creatures come from besides birth: founders, extinction recovery and immigrants."""
import random

from ..config import REPOPULATE_RANDOM_SHARE, RESPAWN_IMMUNITY_TICKS
from ..genetics import combine_brains, combine_genes, mutate_brain, mutate_genes
from ..world.creature import Creature
from ..world.geometry import HEADLESS


def founders(species, world_bounds, count=None, progenitor=None, *, geometry=HEADLESS):
    """A starting population: random brains, or a `progenitor` record (brain, genes,
    generation) followed by mutated copies of it."""
    count = species.start_count if count is None else count
    if progenitor is None:
        return [Creature(species, world_bounds, geometry=geometry) for _ in range(count)]
    genes = progenitor.genes if progenitor.genes is not None else species.default_genes
    population = [Creature(species, world_bounds, brain=progenitor.brain, genes=genes,
                           geometry=geometry)]
    for _ in range(count - 1):
        brain = progenitor.brain.copy()
        mutated = mutate_genes(genes, species)
        mutate_brain(brain, mutated)
        population.append(Creature(species, world_bounds, brain=brain, genes=mutated,
                                   geometry=geometry))
    for c in population:
        c.generation = progenitor.generation
    return population


def repopulate(species, records, world_bounds, *, geometry=HEADLESS):
    """A new population for an extinct species: young of random pairs from the elite pool (or
    the Hall of Fame when the pool is empty), plus REPOPULATE_RANDOM_SHARE random brains."""
    count = species.start_count
    best = records.best
    pool = records.elites or ([best] if best.brain is not None and best.genes else [])
    population = []
    for _ in range(round(count * (1 - REPOPULATE_RANDOM_SHARE)) if pool else 0):
        a, b = random.choice(pool), random.choice(pool)
        genes = combine_genes(a.genes, b.genes, species)
        brain = combine_brains(a.brain, b.brain)
        mutate_brain(brain, genes)
        child = Creature(species, world_bounds, brain=brain, genes=genes, geometry=geometry)
        child.generation = max(a.generation, b.generation) + 1
        population.append(child)
    while len(population) < count:
        population.append(Creature(species, world_bounds, genes=species.default_genes.copy(),
                                   geometry=geometry))
    for c in population:
        c.respawn_immunity_ticks = RESPAWN_IMMUNITY_TICKS
    return population


def immigrants(species, record, count, world_bounds, *, geometry=HEADLESS):
    """`count` creatures carrying the record's brain: one exact copy, the rest mutated."""
    arrivals = []
    for i in range(count):
        brain, genes = record.brain.copy(), dict(record.genes)
        if i:
            genes = mutate_genes(record.genes, species)
            mutate_brain(brain, genes)
        c = Creature(species, world_bounds, brain=brain, genes=genes, geometry=geometry)
        c.respawn_immunity_ticks = RESPAWN_IMMUNITY_TICKS
        c.generation = record.generation
        arrivals.append(c)
    return arrivals
