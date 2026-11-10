"""Compares brain sources by how well their creatures live and breed in seeded headless worlds."""
import random
import time

import numpy as np

from ..persistence.brain_history import resolve
from ..persistence.brain_store import SAVE_DIR
from ..species import ALL_SPECIES
from .bootstrap import start_simulation

BEST, RANDOM = "best", "random"


def source_dir(spec, save_dir=SAVE_DIR):
    """`best`: the saved best brains; `random`: None (random brains); anything else: a history
    snapshot, as --load takes it."""
    if spec == RANDOM:
        return None
    return save_dir if spec == BEST else resolve(spec, save_dir)


def run(source, ticks, seed):
    """One world started from `source` (a folder, or None for random brains), run for `ticks`.
    Returns {species name: {measure: value}} plus "ticks_per_second" under the key "world"."""
    random.seed(seed)
    np.random.seed(seed)
    sim = start_simulation(source)
    world = sim.world
    lived = dict.fromkeys(ALL_SPECIES, 0)
    started = time.perf_counter()
    for _ in range(ticks):
        sim.step()
        for species in ALL_SPECIES:
            lived[species] += len(world.population(species))
    result = {"world": {"ticks_per_second": ticks / (time.perf_counter() - started)}}
    feeding = {False: world.tally.food_eaten, True: world.tally.bites}
    for species in ALL_SPECIES:
        stats = sim.stats[species]
        deaths = sum(stats.deaths.values())
        per_1000 = 1000 / lived[species] if lived[species] else 0.0
        result[species.name] = {
            "population": lived[species] / ticks,
            "extinctions": stats.extinctions,
            "births_per_1000": stats.births * per_1000,
            "litters_per_life": stats.death_litters_total / deaths if deaths else 0.0,
            "age_at_death": stats.death_age_total / deaths if deaths else 0.0,
            "meals_per_1000": feeding[species.is_predator] * per_1000,
        }
    return result


def average(results):
    return {group: {k: sum(r[group][k] for r in results) / len(results) for k in results[0][group]}
            for group in results[0]}


def format_report(rows):
    """rows: [(label, averaged result)]."""
    measures = list(rows[0][1][ALL_SPECIES[0].name])
    lines = []
    for species in ALL_SPECIES:
        lines.append(f"{species.title:<12}" + "".join(f"{m:>18}" for m in measures))
        for label, result in rows:
            lines.append(f"  {label:<10}" + "".join(f"{result[species.name][m]:>18.2f}"
                                                    for m in measures))
    lines.append("ticks/s: " + ", ".join(f"{label} {result['world']['ticks_per_second']:.0f}"
                                         for label, result in rows))
    return "\n".join(lines)
