import random

import numpy as np

from ..brain.batch import BatchBrain
from ..brain.layout import OUTPUTS as OUT
from ..config import (
    ELITE_REINJECT_COUNT,
    ELITE_REINJECT_INTERVAL,
    FOOD_RESPAWN_RATE,
    HOF_SCORE_HALF_LIFE,
    SCORE_SAMPLE_INTERVAL,
)
from ..species import ALL_SPECIES
from ..world.environment import food_cap
from ..world.food import Carcass
from .breeding import breed, mate_seeking_drive
from .populations import immigrants, repopulate
from .records import HallOfFame
from .stats import Statistics


class Simulation:
    """Advances a World tick by tick and runs evolution on it: deaths feed the Hall of Fame,
    extinct species are repopulated from it, and ready adults breed."""

    def __init__(self, world, generation=1, listeners=()):
        self.world = world
        self.generation = generation     # deepest Creature.generation born so far
        self.extinctions = 0
        self.hall_of_fame = HallOfFame()
        self.stats = Statistics()
        self.listeners = list(listeners)
        self._brains = BatchBrain()

    @property
    def tick(self):
        return self.world.tick

    def step(self):
        """One tick. The phase order is part of the rules: reordering changes every run."""
        self.world.advance_clock()
        self._update_food()
        self._update_carcasses()
        self.world.reindex()
        self._think_and_act()
        self._retire_the_dead()
        self._recover_extinct_species()
        if ELITE_REINJECT_INTERVAL and self.tick % ELITE_REINJECT_INTERVAL == 0:
            self._reinject_elites()
        if self.tick % SCORE_SAMPLE_INTERVAL == 0:
            self.hall_of_fame.fade(0.5 ** (SCORE_SAMPLE_INTERVAL / HOF_SCORE_HALF_LIFE))
            self.stats.sample(self.world, self.hall_of_fame)
        self._mark_best()
        self._breed()

    # --- phases ------------------------------------------------------------------------------

    def _update_food(self):
        world = self.world
        for food in world.food:
            food.update()
        world.food[:] = [f for f in world.food if not f.is_rotten() and not f.consumed]
        world.drift_food_patches()
        if random.random() < FOOD_RESPAWN_RATE and len(world.food) < food_cap(world.season):
            world.food.append(world.spawn_food())

    def _update_carcasses(self):
        carcasses = self.world.carcasses
        for carcass in carcasses:
            carcass.update()
        carcasses[:] = [cc for cc in carcasses if not cc.is_gone()]

    def _think_and_act(self):
        world = self.world
        creatures = world.creatures()
        inputs = world.perceive()
        outputs = self._brains.forward(creatures, inputs)
        if len(outputs):
            signals = np.maximum(0.0, outputs[:, OUT.signal]).tolist()
            memories = outputs[:, OUT.memory].tolist()
            for c, sig, mem, x, y in zip(creatures, signals, memories, inputs, outputs):
                c.signal, c.memory = sig, mem
                c.brain_inputs, c.brain_outputs = x, y
        for c, raw in zip(creatures, outputs):
            steered = mate_seeking_drive(c, list(raw), world.index.of(c.species))
            c.mate_override = (steered[OUT.rotation] != raw[OUT.rotation]
                               or steered[OUT.speed] != raw[OUT.speed])
            c.act(steered, world)

    def _retire_the_dead(self):
        world = self.world
        for species in ALL_SPECIES:
            alive = []
            for c in world.population(species):
                if c.is_alive():
                    alive.append(c)
                else:
                    world.carcasses.append(Carcass(c))
                    self.retire(c)
            world.populations[species] = alive
        self._notify("deaths_processed")

    def retire(self, c):
        """Enter a dead creature into the statistics and the Hall of Fame."""
        self.stats[c.species].note_death(c)
        if self.hall_of_fame[c.species].retire(c):
            self._notify("record_set", c.species)

    def _recover_extinct_species(self):
        world = self.world
        for species in ALL_SPECIES:
            if not world.population(species):
                self._notify("extinction", species)
                world.populations[species] = repopulate(
                    species, self.hall_of_fame[species], world.bounds, geometry=world.geometry)
                self.extinctions += 1

    def _reinject_elites(self):
        for species in ALL_SPECIES:
            best = self.hall_of_fame[species].best
            # Never replace a whole small population: the survivors carry the current diversity.
            if (best.brain is not None
                    and len(self.world.population(species)) > 2 * ELITE_REINJECT_COUNT):
                self.replace_weakest(species, best, ELITE_REINJECT_COUNT)

    def _mark_best(self):
        for species in ALL_SPECIES:
            population = self.world.population(species)
            for c in population:
                c.is_best = False
            if population:
                max(population, key=lambda c: c.score).is_best = True

    def _breed(self):
        offspring = breed(self.world)
        for c in offspring:
            self.generation = max(self.generation, c.generation)
        for species in ALL_SPECIES:
            self.world.population(species).extend(c for c in offspring if c.species is species)

    # --- operations for the app ------------------------------------------------------------------

    def replace_weakest(self, species, record, count):
        """Replace the `count` lowest-scoring creatures of the species with immigrants carrying
        the record's brain (see populations.immigrants)."""
        world = self.world
        population = world.population(species)
        replaced = {id(c) for c in sorted(population, key=lambda c: c.score)[:count]}
        arrivals = immigrants(species, record, count, world.bounds, geometry=world.geometry)
        world.populations[species] = [c for c in population if id(c) not in replaced] + arrivals

    def _notify(self, event, *args):
        for listener in self.listeners:
            getattr(listener, event)(self, *args)
