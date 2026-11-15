import math

import numpy as np

from ..brain.batch import BatchBrain
from ..brain.layout import CALL_OUTPUTS
from ..brain.layout import OUTPUTS as OUT
from ..config import (
    ELITE_REINJECT_COUNT,
    ELITE_REINJECT_INTERVAL,
    FOOD_RESPAWN_RATE,
    HOF_SCORE_HALF_LIFE,
    MATE_REFUSE_THRESHOLD,
    SCORE_SAMPLE_INTERVAL,
)
from ..species import ALL_SPECIES
from ..world import creature_table as T
from ..world.environment import food_cap
from ..world.food import Carcass
from . import act_kernel
from .breeding import MateRadar, breed, mate_seeking_drive, stochastic_round
from .populations import immigrants, repopulate
from .records import HallOfFame
from .stats import Statistics


class Simulation:
    """Advances a World tick by tick and runs evolution on it: deaths feed the Hall of Fame,
    extinct species are repopulated from it, and ready adults breed."""

    # Run the act loop compiled (act_kernel) where it supports the world; False runs the Python
    # reference loop.
    compiled_act = True

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
        self.world.separate()
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
        world.age_food()
        world.drift_food_patches()
        room = math.ceil(food_cap(world.season) - len(world.food))
        for _ in range(min(stochastic_round(FOOD_RESPAWN_RATE), room)):
            world.food.append(world.spawn_food())

    def _update_carcasses(self):
        carcasses, tally = self.world.carcasses, self.world.tally
        kept = []
        for carcass in carcasses:
            eaten = carcass.meat <= 0
            carcass.update()
            if not carcass.is_gone():
                kept.append(carcass)
            elif eaten:
                tally.carcasses_eaten += 1
            else:
                tally.carcasses_rotted += 1
        carcasses[:] = kept

    def _think_and_act(self):
        world = self.world
        creatures = world.creatures()
        inputs = world.perceive()
        outputs = self._brains.forward(creatures, inputs)
        world.leave_scent(np.maximum(0.0, outputs[:, OUT.mark]) if len(outputs) else ())
        if len(outputs):
            table = world.table
            calls = np.maximum(0.0, outputs[:, CALL_OUTPUTS])
            if table.holds(creatures):     # World.perceive arranged it
                table.fl[:, T.CALLS] = calls
                table.fl[:, T.SIGNAL] = calls.sum(axis=1)
                table.fl[:, T.MEMORY] = outputs[:, OUT.memory]
                table.brain_io = (inputs, outputs)
            else:
                memories = outputs[:, OUT.memory].tolist()
                for c, call, mem, x, y in zip(creatures, calls, memories, inputs, outputs):
                    c.calls, c.signal, c.memory = call, float(call.sum()), mem
                    c._t.brain_io, c._brain_io = None, (x, y)
            if self.compiled_act and act_kernel.supported(world):
                act_kernel.act_all(world, creatures, outputs)
                return
        self._act_in_python(creatures, outputs)

    def _act_in_python(self, creatures, outputs):
        """The act loop: the mate-seeking drive and Creature.act for each creature in turn. The
        reference for act_kernel.act_all, which must match it exactly."""
        world = self.world
        rows = outputs.tolist() if len(outputs) else []
        radars = {species: MateRadar(world.index.of(species)) for species in ALL_SPECIES}
        for c, row in zip(creatures, rows):
            rotation, speed = row[OUT.rotation], row[OUT.speed]
            c.willing = row[OUT.mate] >= MATE_REFUSE_THRESHOLD
            radar = radars[c.species]
            steered = mate_seeking_drive(c, row, world.index.of(c.species), radar)
            c.mate_override = steered[OUT.rotation] != rotation or steered[OUT.speed] != speed
            c.act(steered, world)
            radar.acted(c)

    def _retire_the_dead(self):
        world = self.world
        table = world.table
        if table.holds(world.creatures()) and (table.fl[:, T.HEALTH] > 0).all():
            self._notify("deaths_processed")    # nobody died
            return
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
        self.stats[c.species].note_death(c, c.death_cause(self.tick))
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
                self.stats[species].extinctions += 1

    def _reinject_elites(self):
        for species in ALL_SPECIES:
            best = self.hall_of_fame[species].best
            # Never replace a whole small population: the survivors carry the current diversity.
            if (best.brain is not None
                    and len(self.world.population(species)) > 2 * ELITE_REINJECT_COUNT):
                self.replace_weakest(species, best, ELITE_REINJECT_COUNT)

    def _mark_best(self):
        world = self.world
        table = world.table
        start = 0 if table.holds(world.creatures()) else None
        for species in ALL_SPECIES:
            population = world.population(species)
            for c in population:
                c.is_best = False
            if not population:
                continue
            if start is None:
                max(population, key=lambda c: c.score).is_best = True
            else:   # np.argmax, like max(), picks the first of equal scores
                end = start + len(population)
                population[int(np.argmax(table.fl[start:end, T.SCORE]))].is_best = True
                start = end

    def _breed(self):
        offspring = breed(self.world)
        for c in offspring:
            self.generation = max(self.generation, c.generation)
            self.stats[c.species].births += 1
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
