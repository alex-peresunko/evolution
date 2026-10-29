import random
import math
import time

import numpy as np

from .config import (
    NUM_HERBIVOROUS, NUM_CARNIVORES, NUM_FOOD, WORLD_WIDTH, WORLD_HEIGHT, GRID_CELL_SIZE,
    DEFAULT_HERBIVORE_GENES, DEFAULT_CARNIVORE_GENES, GENE_MIN_MAX, CARNIVORE_GENE_MIN_MAX,
    HERBIVORE_HEALTH, CARNIVORE_HEALTH, FOOD_RESPAWN_RATE,
    BRAIN_TOPOLOGY, MUTATION_RATE, MUTATION_AMOUNT, AUTOSAVE_INTERVAL,
    MATE_SELECTION_RADIUS, MATE_ATTRACTION_RADIUS, MATE_SEEK_HUNGER_THRESHOLD,
    MATE_SEEK_DANGER_THRESHOLD, MATE_SEEK_BLEND, MATE_SEEK_MIN_SPEED,
    REPRODUCTION_HEALTH_THRESHOLD, MAX_REPRODUCTIONS, REPRODUCTION_COOLDOWN,
    REPRODUCTION_POP_CAP_FACTOR, SMELL_DISTANCE, RESPAWN_IMMUNITY_TICKS,
    SCORE_SAMPLE_INTERVAL,
)
from .entities import Creature, Food, Obstacle
from .spatial import SpatialHashGrid
from .genetics import combine_brains, combine_genes, mutate_genes
from .neural_network import NeuralNetwork
from .brain_store import save_brain, save_brain_if_better


def build_sim_state(world_bounds, creatures, carnivores, food_items, obstacles):
    w, h = world_bounds
    return {
        "running": True,
        "world_bounds": world_bounds,
        "grid": SpatialHashGrid(w, h, GRID_CELL_SIZE),
        "food_grid": SpatialHashGrid(w, h, GRID_CELL_SIZE),
        "herb_grid": SpatialHashGrid(w, h, GRID_CELL_SIZE),
        "carn_grid": SpatialHashGrid(w, h, GRID_CELL_SIZE),
        "creatures": creatures,
        "carnivores": carnivores,
        "food_items": food_items,
        "obstacles": obstacles,
        "herbivore_score_history": [],
        "carnivore_score_history": [],
        "herbivore_longevity_history": [],
        "carnivore_longevity_history": [],
        "max_age_herbivore": 0,
        "max_age_carnivore": 0,
        "generation": 1,
        "tick": 0,
        "last_autosave_tick": 0,
        "hall_of_fame": {
            "herbivore": {"score": -1, "genes": None, "brain": None},
            "carnivore": {"score": -1, "genes": None, "brain": None},
        },
    }


def batch_forward(creatures, inputs_list):
    if not creatures:
        return []
    X = np.array(inputs_list, dtype=np.float64)
    n_layers = len(creatures[0].brain.weights)
    for layer_idx in range(n_layers):
        W = np.stack([c.brain.weights[layer_idx] for c in creatures])
        b = np.stack([c.brain.biases[layer_idx] for c in creatures])
        X = np.einsum("ni,nij->nj", X, W) + b[:, 0, :]
        if layer_idx < n_layers - 1:
            X = np.tanh(X)
    return np.tanh(X)


def _apply_mate_seeking_drive(c, outputs, herb_grid, carn_grid):
    if not c.is_adult:
        return outputs
    if c.hunger >= HUNGER_MAX * MATE_SEEK_HUNGER_THRESHOLD:
        return outputs
    if not c.is_carnivore and c._nearest_smell_strength >= MATE_SEEK_DANGER_THRESHOLD:
        return outputs
    if c.health < c.max_health * REPRODUCTION_HEALTH_THRESHOLD:
        return outputs
    my_grid = carn_grid if c.is_carnivore else herb_grid
    nearby = my_grid.query(c.pos, MATE_ATTRACTION_RADIUS)
    eligible = [
        m for m in nearby
        if m.id != c.id and m.is_adult
        and m.health >= m.max_health * REPRODUCTION_HEALTH_THRESHOLD
        and m.hunger < HUNGER_MAX * MATE_SEEK_HUNGER_THRESHOLD
    ]
    if not eligible:
        return outputs
    nearest = min(eligible, key=lambda m: c.pos.distance_to(m.pos))
    if c.pos.distance_to(nearest.pos) < 1:
        return outputs
    vec = nearest.pos - c.pos
    angle_diff = (math.atan2(vec.y, vec.x) - c.angle + math.pi) % (2 * math.pi) - math.pi
    rotation_drive = angle_diff / math.pi
    outputs[0] = outputs[0] * (1 - MATE_SEEK_BLEND) + rotation_drive * MATE_SEEK_BLEND
    outputs[1] = max(outputs[1], MATE_SEEK_MIN_SPEED)
    return outputs


# Import HUNGER_MAX here to avoid circular; it's already in config
from .config import HUNGER_MAX


def _retire_creature(c, sim_state, prometheus_metrics=None, worker_id=None):
    age_key = "max_age_carnivore" if c.is_carnivore else "max_age_herbivore"
    if c.age > sim_state.get(age_key, 0):
        sim_state[age_key] = c.age
    species = "carnivore" if c.is_carnivore else "herbivore"
    hof = sim_state["hall_of_fame"]
    if c.score > hof[species]["score"]:
        hof[species] = {"score": c.score, "brain": c.brain.copy(), "genes": c.genes.copy()}
        tick = sim_state["tick"]
        if tick - sim_state["last_autosave_tick"] >= AUTOSAVE_INTERVAL:
            # Worker files are per-run scratch; main files must only ever improve.
            if worker_id is not None:
                save = save_brain
                filename = f"worker_{worker_id}_{species}.json"
            else:
                save = save_brain_if_better
                filename = f"best_brain_{species}.json"
            written = save(
                c.brain, filename,
                generation=sim_state["generation"],
                genes=c.genes,
                score=c.score,
                metrics=prometheus_metrics,
            )
            if written is not False:
                sim_state["last_autosave_tick"] = tick


def _repopulate_from_hof(is_carnivore, count, default_genes, sim_state,
                         vec2_factory=None, rect_factory=None):
    species = "carnivore" if is_carnivore else "herbivore"
    constraints = CARNIVORE_GENE_MIN_MAX if is_carnivore else GENE_MIN_MAX
    hof = sim_state["hall_of_fame"][species]
    new_pop = []
    kwargs = {"vec2_factory": vec2_factory, "rect_factory": rect_factory}
    if hof["brain"] and hof["genes"]:
        new_pop.append(Creature(
            sim_state["world_bounds"], brain=hof["brain"].copy(),
            is_carnivore=is_carnivore, genes=hof["genes"].copy(), **kwargs
        ))
        for _ in range((count - 1) // 2):
            b = hof["brain"].copy()
            b.mutate(MUTATION_RATE, MUTATION_AMOUNT)
            g = mutate_genes(hof["genes"], default_genes, gene_constraints=constraints)
            new_pop.append(Creature(
                sim_state["world_bounds"], brain=b,
                is_carnivore=is_carnivore, genes=g, **kwargs
            ))
    while len(new_pop) < count:
        new_pop.append(Creature(
            sim_state["world_bounds"], is_carnivore=is_carnivore,
            genes=default_genes.copy(), **kwargs
        ))
    for c in new_pop:
        c.respawn_immunity_ticks = RESPAWN_IMMUNITY_TICKS
    return new_pop


def update_world(sim_state, prometheus_metrics=None, worker_id=None):
    sim_state["tick"] += 1
    grid = sim_state["grid"]
    for food in sim_state["food_items"]:
        food.update()
    sim_state["food_items"][:] = [
        f for f in sim_state["food_items"] if not f.is_rotten() and not f.consumed
    ]
    if random.random() < FOOD_RESPAWN_RATE and len(sim_state["food_items"]) < NUM_FOOD * 1.5:
        # Inherit vec2 factory from existing food if present, else use default
        vec2_factory = sim_state["food_items"][0]._vec2 if sim_state["food_items"] else None
        sim_state["food_items"].append(Food(sim_state["world_bounds"], vec2_factory=vec2_factory))

    all_creatures = sim_state["creatures"] + sim_state["carnivores"]
    food_grid = sim_state["food_grid"]
    herb_grid = sim_state["herb_grid"]
    carn_grid = sim_state["carn_grid"]
    food_grid.clear()
    herb_grid.clear()
    carn_grid.clear()
    for f in sim_state["food_items"]:
        food_grid.insert(f)
    for h in sim_state["creatures"]:
        herb_grid.insert(h)
    for cn in sim_state["carnivores"]:
        carn_grid.insert(cn)

    all_inputs = []
    for c in all_creatures:
        q_radius = max(c.sight_distance, SMELL_DISTANCE)
        local_food = food_grid.query(c.pos, q_radius)
        local_herbivores = herb_grid.query(c.pos, q_radius)
        local_carnivores = carn_grid.query(c.pos, q_radius)
        all_inputs.append(c.see(local_food, sim_state["obstacles"], local_herbivores, local_carnivores))

    all_outputs = batch_forward(all_creatures, all_inputs)
    for c, outputs in zip(all_creatures, all_outputs):
        outputs = _apply_mate_seeking_drive(c, list(outputs), herb_grid, carn_grid)
        c.act(outputs, sim_state["food_items"], sim_state["creatures"], sim_state["obstacles"], grid,
              food_grid=food_grid, herb_grid=herb_grid)

    alive_herbs, alive_carns = [], []
    for c in sim_state["creatures"]:
        if c.is_alive() and not c.dead:
            alive_herbs.append(c)
        else:
            _retire_creature(c, sim_state, prometheus_metrics, worker_id)
    for c in sim_state["carnivores"]:
        if c.is_alive():
            alive_carns.append(c)
        else:
            _retire_creature(c, sim_state, prometheus_metrics, worker_id)
    sim_state["creatures"] = alive_herbs
    sim_state["carnivores"] = alive_carns

    # Detect extinction and repopulate from Hall of Fame
    _vec2 = sim_state["creatures"][0]._vec2 if sim_state["creatures"] else (
        sim_state["carnivores"][0]._vec2 if sim_state["carnivores"] else None
    )
    _rect = sim_state["creatures"][0]._rect if sim_state["creatures"] else (
        sim_state["carnivores"][0]._rect if sim_state["carnivores"] else None
    )
    for is_carn, key, count, def_genes in [
        (False, "creatures", NUM_HERBIVOROUS, DEFAULT_HERBIVORE_GENES),
        (True, "carnivores", NUM_CARNIVORES, DEFAULT_CARNIVORE_GENES),
    ]:
        if len(sim_state[key]) == 0:
            sp = "carnivore" if is_carn else "herbivore"
            pfx = f"[W{worker_id}] " if worker_id is not None else ""
            print(f"\n{pfx}--- {sp.capitalize()} EXTINCTION tick={sim_state['tick']} gen={sim_state['generation']} ---")
            sim_state[key] = _repopulate_from_hof(
                is_carn, count, def_genes, sim_state,
                vec2_factory=_vec2, rect_factory=_rect,
            )
            sim_state["generation"] += 1

    if sim_state["tick"] % SCORE_SAMPLE_INTERVAL == 0:
        sim_state["herbivore_score_history"].append(sim_state["hall_of_fame"]["herbivore"]["score"])
        sim_state["carnivore_score_history"].append(sim_state["hall_of_fame"]["carnivore"]["score"])
        sim_state["herbivore_longevity_history"].append(sim_state["max_age_herbivore"])
        sim_state["carnivore_longevity_history"].append(sim_state["max_age_carnivore"])

    if sim_state["creatures"]:
        for c in sim_state["creatures"]:
            c.is_best = False
        best_h = max(sim_state["creatures"], key=lambda c: c.score, default=None)
        if best_h:
            best_h.is_best = True
    if sim_state["carnivores"]:
        for c in sim_state["carnivores"]:
            c.is_best = False
        best_c = max(sim_state["carnivores"], key=lambda c: c.score, default=None)
        if best_c:
            best_c.is_best = True

    new_offspring = []
    reproduced_ids = set()
    now = time.time()
    for group, pop_limit, max_health, default_genes in [
        (sim_state["creatures"], NUM_HERBIVOROUS, HERBIVORE_HEALTH, DEFAULT_HERBIVORE_GENES),
        (sim_state["carnivores"], NUM_CARNIVORES, CARNIVORE_HEALTH, DEFAULT_CARNIVORE_GENES),
    ]:
        pop_cap = pop_limit * REPRODUCTION_POP_CAP_FACTOR
        for p1 in group:
            if p1.id in reproduced_ids or len(group) + len(new_offspring) >= pop_cap:
                continue
            p1_is_eligible = (
                p1.is_adult
                and p1.health > max_health * REPRODUCTION_HEALTH_THRESHOLD
                and p1.reproductions < MAX_REPRODUCTIONS
                and (p1.last_reproduction_time is None or now - p1.last_reproduction_time > REPRODUCTION_COOLDOWN)
            )
            if not p1_is_eligible:
                continue
            potential_mates = []
            for p2 in group:
                if p1.id == p2.id or p2.id in reproduced_ids:
                    continue
                p2_is_eligible = (
                    p2.is_adult
                    and p2.health > max_health * REPRODUCTION_HEALTH_THRESHOLD
                    and p2.reproductions < MAX_REPRODUCTIONS
                    and (p2.last_reproduction_time is None or now - p2.last_reproduction_time > REPRODUCTION_COOLDOWN)
                )
                if p2_is_eligible and p1.pos.distance_to(p2.pos) < MATE_SELECTION_RADIUS:
                    potential_mates.append(p2)
            if potential_mates:
                best_mate = max(potential_mates, key=lambda mate: mate.attractiveness)
                constraints = CARNIVORE_GENE_MIN_MAX if p1.is_carnivore else GENE_MIN_MAX
                child_genes = combine_genes(p1.genes, best_mate.genes, default_genes, gene_constraints=constraints)
                child_brain = combine_brains(p1.brain, best_mate.brain)
                child_brain.mutate(MUTATION_RATE, MUTATION_AMOUNT)
                child = Creature(
                    sim_state["world_bounds"], brain=child_brain,
                    is_carnivore=p1.is_carnivore, genes=child_genes,
                    vec2_factory=p1._vec2, rect_factory=p1._rect,
                )
                child.pos = (p1.pos + best_mate.pos) / 2
                child.birth_time = now
                new_offspring.append(child)
                p1.health *= 0.7
                best_mate.health *= 0.7
                p1.reproductions += 1
                best_mate.reproductions += 1
                p1.last_reproduction_time = now
                best_mate.last_reproduction_time = now
                reproduced_ids.add(p1.id)
                reproduced_ids.add(best_mate.id)

    sim_state["creatures"].extend([c for c in new_offspring if not c.is_carnivore])
    sim_state["carnivores"].extend([c for c in new_offspring if c.is_carnivore])


def create_initial_population(count, is_carnivore, world_bounds, brain_file=None,
                               *, vec2_factory=None, rect_factory=None):
    default_genes = DEFAULT_CARNIVORE_GENES if is_carnivore else DEFAULT_HERBIVORE_GENES
    species_name = "carnivore" if is_carnivore else "herbivore"
    constraints = CARNIVORE_GENE_MIN_MAX if is_carnivore else GENE_MIN_MAX
    kwargs = {"vec2_factory": vec2_factory, "rect_factory": rect_factory}
    try:
        if brain_file is None:
            raise FileNotFoundError
        progenitor_brain, gen, progenitor_genes = NeuralNetwork.load(brain_file, BRAIN_TOPOLOGY)
        if progenitor_genes is None:
            progenitor_genes = default_genes
        print(f"Loading initial {species_name} population from {brain_file} (Gen {gen}).")
        new_pop = [Creature(world_bounds, brain=progenitor_brain, is_carnivore=is_carnivore,
                            genes=progenitor_genes, **kwargs)]
        for _ in range(count - 1):
            mutated_brain = progenitor_brain.copy()
            mutated_brain.mutate(MUTATION_RATE, MUTATION_AMOUNT)
            mutated_genes = mutate_genes(progenitor_genes, default_genes, gene_constraints=constraints)
            new_pop.append(Creature(world_bounds, brain=mutated_brain, is_carnivore=is_carnivore,
                                    genes=mutated_genes, **kwargs))
        return new_pop, gen
    except FileNotFoundError:
        print(f"No saved brain found for {species_name}s. Creating a new random population.")
        return [Creature(world_bounds, is_carnivore=is_carnivore, **kwargs) for _ in range(count)], 1


def _fmt_genes(genes):
    if not genes:
        return "n/a"
    g = genes
    return (
        f"spd:{g.get('max_speed', 0):.2f}  sz:{g.get('size', 0):.1f}"
        f"  sight:{g.get('sight_distance', 0):.0f}  stam:{g.get('max_stamina', 0):.0f}"
    )
