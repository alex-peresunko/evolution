import random
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from prometheus_client import start_http_server

from evolution import config
from evolution.entities import Creature, Food, Obstacle
from evolution.math_types import Vector2
from evolution.spatial import SpatialHashGrid
from evolution.simulation import create_initial_population, _apply_mate_seeking_drive
from evolution.genetics import combine_brains, combine_genes, mutate_genes
from evolution.metrics import GameMetrics
from evolution.brain_store import read_saved_score, save_brain, save_brain_if_better

# =============================================================================
# IPC-only lean dataclasses (not part of the shared library)
# =============================================================================
@dataclass(frozen=True)
class LeanCreature:
    id: int
    pos: object
    size: float
    is_carnivore: bool
    angle: float
    current_speed: float
    max_speed: float
    signal: float


@dataclass(frozen=True)
class LeanFood:
    id: int
    pos: object
    health_value: int


# =============================================================================
# Parallel worker function
# =============================================================================
def process_creature_chunk(args):
    creature_chunk, all_herbivores, all_carnivores, all_food, obstacles, world_bounds = args

    lean_herbivores = [LeanCreature(c.id, c.pos, c.size, c.is_carnivore, c.angle, c.current_speed, c.max_speed, c.signal) for c in all_herbivores]
    lean_carnivores = [LeanCreature(c.id, c.pos, c.size, c.is_carnivore, c.angle, c.current_speed, c.max_speed, c.signal) for c in all_carnivores]
    lean_food = [LeanFood(f.id, f.pos, f.get_health_value()) for f in all_food]

    food_eaten_ids = set()
    prey_eaten_ids = set()

    grid = SpatialHashGrid(world_bounds[0], world_bounds[1], config.GRID_CELL_SIZE)
    for obj in lean_herbivores + lean_carnivores + lean_food:
        grid.insert(obj)

    herb_grid_full = SpatialHashGrid(world_bounds[0], world_bounds[1], config.GRID_CELL_SIZE)
    carn_grid_full = SpatialHashGrid(world_bounds[0], world_bounds[1], config.GRID_CELL_SIZE)
    for h in all_herbivores:
        herb_grid_full.insert(h)
    for cn in all_carnivores:
        carn_grid_full.insert(cn)

    lean_food_health_map = {f.id: f.health_value for f in lean_food}

    for c in creature_chunk:
        local_objects = grid.query(c.pos, max(c.sight_distance, config.SMELL_DISTANCE))
        local_food = [obj for obj in local_objects if isinstance(obj, LeanFood)]
        local_herbivores = [obj for obj in local_objects if isinstance(obj, LeanCreature) and not obj.is_carnivore]
        local_carnivores = [obj for obj in local_objects if isinstance(obj, LeanCreature) and obj.is_carnivore]

        outputs = c.perceive_and_think(local_food, local_herbivores, local_carnivores, obstacles)
        c.signal = max(0.0, float(outputs[2]))
        c.memory = [float(m) for m in outputs[3:3 + config.NUM_MEMORY]]
        outputs = _apply_mate_seeking_drive(c, list(outputs), herb_grid_full, carn_grid_full)

        interaction_radius = c.size + config.GENE_MIN_MAX["size"][1]
        nearby_objects = grid.query(c.pos, interaction_radius)

        if c.is_carnivore and c.respawn_immunity_ticks == 0:
            base_size = config.DEFAULT_CARNIVORE_GENES["size"]
            for obj in nearby_objects:
                if isinstance(obj, LeanCreature) and not obj.is_carnivore:
                    if obj.id in prey_eaten_ids:
                        continue
                    if c.pos.distance_to(obj.pos) < (c.size + obj.size):
                        prey_eaten_ids.add(obj.id)
                        base_health_gain = config.CARNIVORE_HEALTH_PER_FOOD
                        size_penalty = 1 + (c.genetic_size - base_size) * config.HEALTH_GAIN_SIZE_PENALTY
                        c.health = min(c.max_health, c.health + base_health_gain / max(0.1, size_penalty))
                        c.score += 1
                        break
        else:
            base_size = config.DEFAULT_HERBIVORE_GENES["size"]
            for obj in nearby_objects:
                if isinstance(obj, LeanFood):
                    if obj.id in food_eaten_ids:
                        continue
                    if c.pos.distance_to(obj.pos) < (c.size + config.FOOD_RADIUS):
                        food_eaten_ids.add(obj.id)
                        base_health_gain = lean_food_health_map.get(obj.id, 0)
                        size_penalty = 1 + (c.genetic_size - base_size) * config.HEALTH_GAIN_SIZE_PENALTY
                        c.health = min(c.max_health, c.health + base_health_gain / max(0.1, size_penalty))
                        c.score += 1
                        break

        # Eating interactions already handled above; call act() for movement + obstacle collision.
        # Pass empty grids so the act() eating logic finds nothing (already consumed).
        _empty = SpatialHashGrid(world_bounds[0], world_bounds[1], config.GRID_CELL_SIZE)
        c.act(outputs, [], [], obstacles, _empty)

    return creature_chunk, food_eaten_ids, prey_eaten_ids


# =============================================================================
# Multiproc update_world (overrides shared version)
# =============================================================================
def update_world(sim_state, executor):
    sim_state["generation_timer"] += 1
    sim_state["tick"] += 1

    for food in sim_state["food_items"]:
        food.update()
    sim_state["food_items"][:] = [f for f in sim_state["food_items"] if not f.is_rotten()]
    if random.random() < config.FOOD_RESPAWN_RATE and len(sim_state["food_items"]) < config.NUM_FOOD * 1.5:
        sim_state["food_items"].append(Food(sim_state["world_bounds"]))

    all_creatures = sim_state["creatures"] + sim_state["carnivores"]
    if not all_creatures:
        return

    num_workers = executor._max_workers
    chunks = np.array_split(all_creatures, num_workers)

    task_args = [
        (
            chunk.tolist(),
            sim_state["creatures"],
            sim_state["carnivores"],
            sim_state["food_items"],
            sim_state["obstacles"],
            sim_state["world_bounds"],
        )
        for chunk in chunks if chunk.size > 0
    ]

    results = executor.map(process_creature_chunk, task_args)

    updated_creatures = []
    total_food_eaten_ids = set()
    total_prey_eaten_ids = set()
    for chunk, food_ids, prey_ids in results:
        updated_creatures.extend(chunk)
        total_food_eaten_ids.update(food_ids)
        total_prey_eaten_ids.update(prey_ids)

    sim_state["food_items"] = [f for f in sim_state["food_items"] if f.id not in total_food_eaten_ids]

    final_herbivores = []
    final_carnivores = []
    for c in updated_creatures:
        if c.is_alive():
            if c.is_carnivore:
                final_carnivores.append(c)
            elif c.id not in total_prey_eaten_ids:
                final_herbivores.append(c)

    sim_state["creatures"] = final_herbivores
    sim_state["carnivores"] = final_carnivores

    new_offspring = []
    reproduced_ids = set()
    tick = sim_state["tick"]
    now = time.time()
    for group, pop_limit, max_health, default_genes in [
        (sim_state["creatures"], config.NUM_HERBIVOROUS, config.HERBIVORE_HEALTH, config.DEFAULT_HERBIVORE_GENES),
        (sim_state["carnivores"], config.NUM_CARNIVORES, config.CARNIVORE_HEALTH, config.DEFAULT_CARNIVORE_GENES),
    ]:
        pop_cap = pop_limit * config.REPRODUCTION_POP_CAP_FACTOR
        for p1 in group:
            if p1.id in reproduced_ids or len(group) + len(new_offspring) >= pop_cap:
                continue
            p1_is_eligible = (
                p1.is_adult
                and p1.health > max_health * config.REPRODUCTION_HEALTH_THRESHOLD
                and p1.reproductions < config.MAX_REPRODUCTIONS
                and (p1.last_reproduction_tick is None
                     or tick - p1.last_reproduction_tick > config.REPRODUCTION_COOLDOWN_TICKS)
            )
            if not p1_is_eligible:
                continue
            potential_mates = [
                p2 for p2 in group
                if p1.id != p2.id and p2.id not in reproduced_ids
                and p2.is_adult
                and p2.health > max_health * config.REPRODUCTION_HEALTH_THRESHOLD
                and p2.reproductions < config.MAX_REPRODUCTIONS
                and (p2.last_reproduction_tick is None
                     or tick - p2.last_reproduction_tick > config.REPRODUCTION_COOLDOWN_TICKS)
                and p1.pos.distance_to(p2.pos) < config.MATE_SELECTION_RADIUS
            ]
            if potential_mates:
                best_mate = max(potential_mates, key=lambda mate: mate.attractiveness)
                child_genes = combine_genes(p1.genes, best_mate.genes, default_genes)
                child_brain = combine_brains(p1.brain, best_mate.brain)
                child = Creature(sim_state["world_bounds"], brain=child_brain,
                                 is_carnivore=p1.is_carnivore, genes=child_genes)
                child.pos = (p1.pos + best_mate.pos) / 2
                child.birth_time = now
                new_offspring.append(child)
                p1.health *= 0.7
                best_mate.health *= 0.7
                p1.reproductions += 1
                best_mate.reproductions += 1
                p1.last_reproduction_tick = tick
                best_mate.last_reproduction_tick = tick
                reproduced_ids.add(p1.id)
                reproduced_ids.add(best_mate.id)

    sim_state["creatures"].extend([c for c in new_offspring if not c.is_carnivore])
    sim_state["carnivores"].extend([c for c in new_offspring if c.is_carnivore])


def evolve_population(sim_state, prometheus_metrics):
    sim_state["herbivore_score_history"].append(sim_state["hall_of_fame"]["herbivore"]["score"])
    sim_state["carnivore_score_history"].append(sim_state["hall_of_fame"]["carnivore"]["score"])
    hall_of_fame = sim_state["hall_of_fame"]

    best_h = max(sim_state["creatures"], key=lambda c: c.score, default=None)
    if best_h and best_h.score > hall_of_fame["herbivore"]["score"]:
        print(f"\nNew Herbivore record! Gen {sim_state['generation']}, Score: {best_h.score}")
        hall_of_fame["herbivore"] = {"score": best_h.score, "genes": best_h.genes.copy(),
                                     "brain": best_h.brain.copy()}
    best_c = max(sim_state["carnivores"], key=lambda c: c.score, default=None)
    if best_c and best_c.score > hall_of_fame["carnivore"]["score"]:
        print(f"\nNew Carnivore record! Gen {sim_state['generation']}, Score: {best_c.score}")
        hall_of_fame["carnivore"] = {"score": best_c.score, "genes": best_c.genes.copy(),
                                     "brain": best_c.brain.copy()}

    def get_new_population(survivors, count, is_carnivore, default_genes):
        species = "carnivore" if is_carnivore else "herbivore"
        if not survivors:
            print(f"\n--- {species.capitalize()} extinction! Repopulating from Hall of Fame. ---")
            progenitor_record = hall_of_fame[species]
            new_pop = []
            if progenitor_record["genes"] and progenitor_record["brain"]:
                progenitor = Creature(
                    sim_state["world_bounds"], brain=progenitor_record["brain"].copy(),
                    is_carnivore=is_carnivore, genes=progenitor_record["genes"].copy()
                )
                new_pop.append(progenitor)
                for _ in range((count - 1) // 2):
                    b = progenitor_record["brain"].copy()
                    b.mutate(config.MUTATION_RATE, config.MUTATION_AMOUNT)
                    g = mutate_genes(progenitor_record["genes"], default_genes)
                    new_pop.append(Creature(sim_state["world_bounds"], brain=b,
                                            is_carnivore=is_carnivore, genes=g))
            while len(new_pop) < count:
                new_pop.append(Creature(sim_state["world_bounds"], is_carnivore=is_carnivore,
                                        genes=default_genes.copy()))
            for c in new_pop:
                c.respawn_immunity_ticks = config.RESPAWN_IMMUNITY_TICKS
            return new_pop
        new_pop = []
        for _ in range(count):
            parent1 = random.choice(survivors)
            if len(survivors) > 1 and random.random() < 0.7:
                parent2 = random.choice(survivors)
                new_brain = combine_brains(parent1.brain, parent2.brain)
            else:
                new_brain = parent1.brain.copy()
            new_brain.mutate(config.MUTATION_RATE, config.MUTATION_AMOUNT)
            new_genes = mutate_genes(parent1.genes, default_genes)
            new_pop.append(Creature(sim_state["world_bounds"], brain=new_brain,
                                    is_carnivore=is_carnivore, genes=new_genes))
        return new_pop

    sim_state["creatures"].sort(key=lambda c: c.score, reverse=True)
    num_to_select_h = max(1, int(len(sim_state["creatures"]) * config.SURVIVAL_RATE)) if sim_state["creatures"] else 0
    survivors_h = sim_state["creatures"][:num_to_select_h]
    sim_state["carnivores"].sort(key=lambda c: c.score, reverse=True)
    num_to_select_c = max(1, int(len(sim_state["carnivores"]) * config.SURVIVAL_RATE)) if sim_state["carnivores"] else 0
    survivors_c = sim_state["carnivores"][:num_to_select_c]

    if sim_state["generation"] % config.AUTOSAVE_INTERVAL == 0:
        print(f"\n--- AUTOSAVING BRAINS FOR END OF GENERATION {sim_state['generation']} ---")
        if survivors_h:
            save_brain(survivors_h[0].brain,
                       f"autosave_herbivore_gen_{sim_state['generation']}.json",
                       generation=sim_state["generation"], genes=survivors_h[0].genes,
                       score=survivors_h[0].score, metrics=prometheus_metrics)
        if survivors_c:
            save_brain(survivors_c[0].brain,
                       f"autosave_carnivore_gen_{sim_state['generation']}.json",
                       generation=sim_state["generation"], genes=survivors_c[0].genes,
                       score=survivors_c[0].score, metrics=prometheus_metrics)

    sim_state["creatures"] = get_new_population(survivors_h, config.NUM_HERBIVOROUS,
                                                 False, config.DEFAULT_HERBIVORE_GENES)
    sim_state["carnivores"] = get_new_population(survivors_c, config.NUM_CARNIVORES,
                                                  True, config.DEFAULT_CARNIVORE_GENES)
    sim_state["generation"] += 1
    sim_state["generation_timer"] = 0


# =============================================================================
# Main
# =============================================================================
def main():
    world_bounds = (config.WORLD_WIDTH, config.WORLD_HEIGHT)
    herbivores, h_gen = create_initial_population(
        config.NUM_HERBIVOROUS, False, world_bounds, "best_brain_herbivore.json"
    )
    carnivores, c_gen = create_initial_population(
        config.NUM_CARNIVORES, True, world_bounds, "best_brain_carnivore.json"
    )
    sim_state = {
        "running": True,
        "world_bounds": world_bounds,
        "creatures": herbivores,
        "carnivores": carnivores,
        "food_items": [Food(world_bounds) for _ in range(config.NUM_FOOD)],
        "obstacles": [
            Obstacle(
                random.randint(100, config.WORLD_WIDTH - 100),
                random.randint(100, config.WORLD_HEIGHT - 100),
                random.randint(50, 150),
                random.randint(50, 150),
            )
            for _ in range(config.NUM_OBSTACLES)
        ],
        "herbivore_score_history": [],
        "carnivore_score_history": [],
        "generation": max(h_gen, c_gen),
        "generation_timer": 0,
        "tick": 0,
        "hall_of_fame": {
            "herbivore": {"score": -1, "genes": None, "brain": None},
            "carnivore": {"score": -1, "genes": None, "brain": None},
        },
    }
    prometheus_metrics = GameMetrics(sim_state)
    last_time = time.time()
    ticks_this_second = 0
    last_tps_report_time = time.time()

    GENERATION_TIME = 600
    num_workers = cpu_count()
    print(f"--- Starting Headless Multi-Core Simulation ({num_workers} workers) ---")
    print(f"Starting at Generation {sim_state['generation']}. Press Ctrl+C to stop.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        try:
            while sim_state["running"]:
                now = time.time()
                dt = now - last_time
                last_time = now

                update_world(sim_state, executor)

                if sim_state["generation_timer"] > GENERATION_TIME:
                    evolve_population(sim_state, prometheus_metrics)
                    print(f"\n--- Generation {sim_state['generation'] - 1} complete ---")
                    print(f"    HoF Herbivore: {sim_state['hall_of_fame']['herbivore']['score']}")
                    print(f"    HoF Carnivore: {sim_state['hall_of_fame']['carnivore']['score']}")

                prometheus_metrics.update(sim_state, dt)
                ticks_this_second += 1
                if now - last_tps_report_time >= 1.0:
                    tps = ticks_this_second
                    ticks_this_second = 0
                    last_tps_report_time = now
                    progress = sim_state["generation_timer"] / GENERATION_TIME
                    bar = ("#" * int(progress * 20)).ljust(20, "-")
                    sys.stdout.write(
                        f"\rGen {sim_state['generation']} [{bar}] {int(progress * 100)}%"
                        f" | TPS: {tps}"
                        f" | Herb: {len(sim_state['creatures'])}"
                        f" | Carn: {len(sim_state['carnivores'])}   "
                    )
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")
        finally:
            print("\nSaving best brains from Hall of Fame on exit...")
            for species in ("herbivore", "carnivore"):
                hof = sim_state["hall_of_fame"][species]
                filename = f"best_brain_{species}.json"
                if hof["brain"] is None:
                    print(f"No {species} in Hall of Fame to save.")
                elif not save_brain_if_better(hof["brain"], filename,
                                              generation=sim_state["generation"],
                                              genes=hof["genes"], score=hof["score"],
                                              metrics=prometheus_metrics):
                    print(f"Kept existing {filename} (saved score "
                          f"{read_saved_score(filename):.3f} >= {hof['score']:.3f})")
            print("Exiting.")


if __name__ == "__main__":
    start_http_server(8000)
    main()
