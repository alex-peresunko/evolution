import random
import math
import time
from collections import deque

import numpy as np

from .config import (
    NUM_HERBIVOROUS, NUM_CARNIVORES, NUM_FOOD, FOOD_RADIUS, WORLD_WIDTH, WORLD_HEIGHT, GRID_CELL_SIZE,
    DEFAULT_HERBIVORE_GENES, DEFAULT_CARNIVORE_GENES, GENE_MIN_MAX, CARNIVORE_GENE_MIN_MAX,
    HERBIVORE_HEALTH, CARNIVORE_HEALTH, FOOD_RESPAWN_RATE,
    BRAIN_TOPOLOGY, AUTOSAVE_INTERVAL,
    MATE_SELECTION_RADIUS, MATE_ATTRACTION_RADIUS, MATE_SEEK_HUNGER_THRESHOLD,
    MATE_SEEK_DANGER_THRESHOLD, MATE_SEEK_BLEND, MATE_SEEK_MIN_SPEED,
    REPRODUCTION_HEALTH_THRESHOLD, MAX_REPRODUCTIONS, REPRODUCTION_COOLDOWN_TICKS,
    REPRODUCTION_POP_CAP_FACTOR, RESPAWN_IMMUNITY_TICKS,
    ELITE_REINJECT_INTERVAL, ELITE_REINJECT_COUNT, MATE_SCORE_WEIGHT,
    LITTER_ENDOWMENT, LITTER_SPREAD,
    SCORE_SAMPLE_INTERVAL, CHART_HISTORY_SAVE_MAX, NUM_MEMORY, DAY_LENGTH_TICKS,
    SEASON_LENGTH_TICKS, SEASON_FOOD_AMPLITUDE,
    FOOD_PATCH_COUNT, FOOD_PATCH_SHARE, FOOD_PATCH_RADIUS, FOOD_PATCH_SPEED, FOOD_PATCH_TURN,
    NUM_LAKES, LAKE_RADIUS_RANGE,
)
from .entities import Carcass, Creature, Food, Lake, Obstacle
from .spatial import SpatialHashGrid
from .genetics import combine_brains, combine_genes, compatible, mutate_brain, mutate_genes
from .islands import inject_immigrants
from .neural_network import NeuralNetwork
from .perception import perceive_all
from .brain_store import (read_saved_history, read_saved_score, save_brain, save_brain_if_better,
                          update_saved_history)


def generate_lakes(world_bounds, obstacles=(), count=NUM_LAKES, attempts=50):
    """Up to `count` random lakes inside the walls that don't touch an obstacle."""
    w, h = world_bounds
    lakes = []
    for _ in range(count * attempts):
        if len(lakes) == count:
            break
        r = min(random.uniform(*LAKE_RADIUS_RANGE), w / 2, h / 2)
        x, y = random.uniform(r, w - r), random.uniform(r, h - r)
        if not any(_circle_hits_rect(x, y, r, obs.rect) for obs in obstacles):
            lakes.append(Lake(x, y, r))
    return lakes


def _circle_hits_rect(x, y, r, rect):
    nx = min(max(x, rect.left), rect.right)
    ny = min(max(y, rect.top), rect.bottom)
    return (x - nx) ** 2 + (y - ny) ** 2 < r ** 2


def build_sim_state(world_bounds, creatures, carnivores, food_items, obstacles, lakes=()):
    w, h = world_bounds
    return {
        "running": True,
        "world_bounds": world_bounds,
        "grid": SpatialHashGrid(w, h, GRID_CELL_SIZE),
        "food_grid": SpatialHashGrid(w, h, GRID_CELL_SIZE),
        "herb_grid": SpatialHashGrid(w, h, GRID_CELL_SIZE),
        "carn_grid": SpatialHashGrid(w, h, GRID_CELL_SIZE),
        "carcass_grid": SpatialHashGrid(w, h, GRID_CELL_SIZE),
        "carcasses": [],
        "food_patches": [
            {"x": random.uniform(0, w), "y": random.uniform(0, h),
             "heading": random.uniform(0, 2 * math.pi)}
            for _ in range(FOOD_PATCH_COUNT)
        ],
        "light": 1.0,
        "season": 0.0,
        "creatures": creatures,
        "carnivores": carnivores,
        "food_items": food_items,
        "obstacles": obstacles,
        "lakes": list(lakes),
        "herbivore_score_history": [],
        "carnivore_score_history": [],
        "herbivore_longevity_history": [],
        "carnivore_longevity_history": [],
        "max_age_herbivore": 0,
        "max_age_carnivore": 0,
        "gene_history": {"herbivore": {}, "carnivore": {}},
        "lineages": {"herbivore": 0, "carnivore": 0},
        "generation": 1,
        "tick": 0,
        "last_autosave_tick": 0,
        "hof_unsaved": set(),
        "hall_of_fame": {
            "herbivore": {"score": -1, "genes": None, "brain": None},
            "carnivore": {"score": -1, "genes": None, "brain": None},
        },
    }


_stacked_weights = {"key": None, "layers": None}


def _stacked_layers(creatures):
    """Per-layer (W, b) stacked across creatures; rebuilt only when the population changes
    (a creature's brain never changes after it is created)."""
    key = tuple((c.id, id(c.brain)) for c in creatures)
    if _stacked_weights["key"] != key:
        n_layers = len(creatures[0].brain.weights)
        _stacked_weights["layers"] = [
            (np.stack([c.brain.weights[i] for c in creatures]),
             np.stack([c.brain.biases[i] for c in creatures])[:, 0, :])
            for i in range(n_layers)
        ]
        _stacked_weights["key"] = key
    return _stacked_weights["layers"]


def batch_forward(creatures, inputs_list):
    if not creatures:
        return []
    X = np.asarray(inputs_list, dtype=np.float64)
    layers = _stacked_layers(creatures)
    n_layers = len(layers)
    for layer_idx, (W, b) in enumerate(layers):
        X = np.einsum("ni,nij->nj", X, W) + b
        if layer_idx < n_layers - 1:
            X = np.tanh(X)
    return np.tanh(X)


def day_light(tick):
    """0 at midnight, 1 at noon; a run starts at noon."""
    return 0.5 + 0.5 * math.cos(2 * math.pi * tick / DAY_LENGTH_TICKS)


def season(tick):
    """-1 at midwinter, 1 at midsummer; a run starts in spring (0, rising)."""
    return math.sin(2 * math.pi * tick / SEASON_LENGTH_TICKS)


def season_name(tick):
    phase = (tick / SEASON_LENGTH_TICKS) % 1
    return ("Spring", "Summer", "Autumn", "Winter")[int((phase + 0.125) * 4) % 4]


def food_cap(season_value):
    """Most food items on the map at once: plentiful summers, lean winters."""
    return NUM_FOOD * 1.5 * (1 + SEASON_FOOD_AMPLITUDE * season_value)


def _update_food_patches(sim_state):
    w, h = sim_state["world_bounds"]
    for patch in sim_state["food_patches"]:
        patch["heading"] += random.uniform(-FOOD_PATCH_TURN, FOOD_PATCH_TURN)
        x = patch["x"] + math.cos(patch["heading"]) * FOOD_PATCH_SPEED
        y = patch["y"] + math.sin(patch["heading"]) * FOOD_PATCH_SPEED
        if not 0 <= x <= w:  # bounce off the world edges
            patch["heading"] = math.pi - patch["heading"]
        if not 0 <= y <= h:
            patch["heading"] = -patch["heading"]
        patch["x"], patch["y"] = min(max(x, 0), w), min(max(y, 0), h)


def _spawn_food(sim_state, attempts=10):
    """New food goes near a random patch (FOOD_PATCH_SHARE of the time) or anywhere, on dry
    land unless `attempts` placements all land in a lake."""
    lakes = sim_state.get("lakes", ())
    for _ in range(attempts):
        food = _place_food(sim_state)
        if not any(lake.contains(food.pos.x, food.pos.y) for lake in lakes):
            break
    return food


def _place_food(sim_state):
    w, h = sim_state["world_bounds"]
    # Inherit vec2 factory from existing food if present, else use default
    vec2_factory = sim_state["food_items"][0]._vec2 if sim_state["food_items"] else None
    pos = None
    patches = sim_state.get("food_patches")
    if patches and random.random() < FOOD_PATCH_SHARE:
        patch = random.choice(patches)
        pos = (min(max(patch["x"] + random.gauss(0, FOOD_PATCH_RADIUS), FOOD_RADIUS),
                   w - FOOD_RADIUS),
               min(max(patch["y"] + random.gauss(0, FOOD_PATCH_RADIUS), FOOD_RADIUS),
                   h - FOOD_RADIUS))
    return Food(sim_state["world_bounds"], vec2_factory=vec2_factory, pos=pos)


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
    constraints = CARNIVORE_GENE_MIN_MAX if c.is_carnivore else GENE_MIN_MAX
    nearby = my_grid.query(c.pos, MATE_ATTRACTION_RADIUS)
    eligible = [
        m for m in nearby
        if m.id != c.id and m.is_adult
        and m.health >= m.max_health * REPRODUCTION_HEALTH_THRESHOLD
        and m.hunger < HUNGER_MAX * MATE_SEEK_HUNGER_THRESHOLD
        and compatible(c.genes, m.genes, constraints)
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


def species_history(sim_state, species):
    """One species' chart histories, trimmed for saving into its brain file."""
    return {metric: list(sim_state[f"{species}_{metric}_history"][-CHART_HISTORY_SAVE_MAX:])
            for metric in ("score", "longevity")}


def restore_progress(sim_state, species, brain, genes, filename, save_dir="saved_brains"):
    """After loading a saved brain, continue where its run left off: seed the Hall of Fame with
    the saved score (so new chart samples don't drop to 0), and restore the chart histories."""
    score = read_saved_score(filename, save_dir=save_dir)
    if score is not None:
        sim_state["hall_of_fame"][species] = {"score": score, "brain": brain.copy(),
                                              "genes": dict(genes or {})}
    history = read_saved_history(filename, save_dir=save_dir) or {}
    for metric in ("score", "longevity"):
        sim_state[f"{species}_{metric}_history"] = list(history.get(metric, []))
    longevity = sim_state[f"{species}_longevity_history"]
    if longevity:
        sim_state[f"max_age_{species}"] = max(sim_state[f"max_age_{species}"], int(longevity[-1]))
    # Both species share the chart x axis: left-pad the shorter histories to equal length.
    keys = [f"{sp}_{m}_history" for sp in ("herbivore", "carnivore") for m in ("score", "longevity")]
    length = max(len(sim_state[k]) for k in keys)
    for k in keys:
        h = sim_state[k]
        sim_state[k] = [h[0] if h else 0] * (length - len(h)) + h


def _stochastic_round(x):
    """floor(x), plus one with probability frac(x): 1.3 gives 2 young 30% of the time."""
    n = int(x)
    return n + (random.random() < x - n)


def sample_genes(sim_state):
    """Append each species' mean gene values to gene_history and count its live lineages."""
    for species, key, defaults in (("herbivore", "creatures", DEFAULT_HERBIVORE_GENES),
                                   ("carnivore", "carnivores", DEFAULT_CARNIVORE_GENES)):
        group = sim_state[key]
        history = sim_state.setdefault("gene_history", {}).setdefault(species, {})
        for gene in defaults:
            series = history.setdefault(gene, deque(maxlen=CHART_HISTORY_SAVE_MAX))
            if group:
                series.append(sum(c.genes[gene] for c in group) / len(group))
        sim_state.setdefault("lineages", {})[species] = len({c.lineage for c in group})


def _retire_creature(c, sim_state, prometheus_metrics=None, worker_id=None):
    age_key = "max_age_carnivore" if c.is_carnivore else "max_age_herbivore"
    if c.age > sim_state.get(age_key, 0):
        sim_state[age_key] = c.age
    species = "carnivore" if c.is_carnivore else "herbivore"
    hof = sim_state["hall_of_fame"]
    if c.score > hof[species]["score"]:
        hof[species] = {"score": c.score, "brain": c.brain.copy(), "genes": c.genes.copy()}
        sim_state.setdefault("hof_unsaved", set()).add(species)
        autosave_hall_of_fame(sim_state, prometheus_metrics, worker_id)


def autosave_hall_of_fame(sim_state, prometheus_metrics=None, worker_id=None, force=False):
    """Write Hall of Fame records not saved yet, at most once per AUTOSAVE_INTERVAL ticks
    (unless force). A record set inside the interval waits for the next call instead of being
    dropped. Worker files are per-run scratch; main files must only ever improve."""
    unsaved = sim_state.get("hof_unsaved")
    tick = sim_state["tick"]
    if not unsaved or (not force and tick - sim_state["last_autosave_tick"] < AUTOSAVE_INTERVAL):
        return
    for species in sorted(unsaved):
        hof = sim_state["hall_of_fame"][species]
        if worker_id is not None:
            save, filename = save_brain, f"worker_{worker_id}_{species}.json"
        else:
            save, filename = save_brain_if_better, f"best_brain_{species}.json"
        written = save(hof["brain"], filename, generation=sim_state["generation"],
                       genes=hof["genes"], score=hof["score"], metrics=prometheus_metrics,
                       history=species_history(sim_state, species))
        if written is not False:
            sim_state["last_autosave_tick"] = tick
    unsaved.clear()


def save_hall_of_fame(sim_state, prometheus_metrics=None):
    """Exit save: each species' Hall of Fame into best_brain_*.json if it beats the saved score,
    otherwise only the file's chart history is updated."""
    for species in ("herbivore", "carnivore"):
        hof = sim_state["hall_of_fame"][species]
        filename = f"best_brain_{species}.json"
        history = species_history(sim_state, species)
        if hof["brain"] is None:
            print(f"No {species} in Hall of Fame to save.")
        elif not save_brain_if_better(hof["brain"], filename,
                                      generation=sim_state["generation"], genes=hof["genes"],
                                      score=hof["score"], metrics=prometheus_metrics,
                                      history=history):
            update_saved_history(filename, history)
            print(f"Kept existing {filename} (saved score "
                  f"{read_saved_score(filename):.3f} >= {hof['score']:.3f}), updated its chart history")


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
            g = mutate_genes(hof["genes"], default_genes, gene_constraints=constraints)
            mutate_brain(b, g)
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


def _reinject_elites(sim_state):
    for species, key in (("herbivore", "creatures"), ("carnivore", "carnivores")):
        hof = sim_state["hall_of_fame"][species]
        # Never replace a whole small population: the survivors carry the current diversity.
        if hof["brain"] is not None and len(sim_state[key]) > 2 * ELITE_REINJECT_COUNT:
            inject_immigrants(sim_state, species, hof["brain"], hof["genes"],
                              count=ELITE_REINJECT_COUNT)


def _mate_preference(mates, default_genes):
    top_score = max((m.score for m in mates), default=0) or 1.0
    return lambda m: (m.attractiveness / default_genes["attractiveness"]
                      + MATE_SCORE_WEIGHT * m.score / top_score)


def update_world(sim_state, prometheus_metrics=None, worker_id=None):
    sim_state["tick"] += 1
    sim_state["light"] = light = day_light(sim_state["tick"])
    sim_state["season"] = season_value = season(sim_state["tick"])
    grid = sim_state["grid"]
    for food in sim_state["food_items"]:
        food.update()
    sim_state["food_items"][:] = [
        f for f in sim_state["food_items"] if not f.is_rotten() and not f.consumed
    ]
    if "food_patches" in sim_state:
        _update_food_patches(sim_state)
    if random.random() < FOOD_RESPAWN_RATE and len(sim_state["food_items"]) < food_cap(season_value):
        sim_state["food_items"].append(_spawn_food(sim_state))
    carcasses = sim_state.setdefault("carcasses", [])
    for carcass in carcasses:
        carcass.update()
    carcasses[:] = [cc for cc in carcasses if not cc.is_gone()]

    all_creatures = sim_state["creatures"] + sim_state["carnivores"]
    lakes = sim_state.get("lakes", ())
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
    carcass_grid = sim_state.setdefault("carcass_grid", SpatialHashGrid(
        *sim_state["world_bounds"], GRID_CELL_SIZE))
    carcass_grid.clear()
    for cc in carcasses:
        carcass_grid.insert(cc)

    all_inputs = perceive_all(all_creatures, sim_state["food_items"], sim_state["creatures"],
                              sim_state["carnivores"], sim_state["obstacles"], carcasses, light,
                              season_value, lakes)
    all_outputs = batch_forward(all_creatures, all_inputs)
    if len(all_outputs):
        signals = np.maximum(0.0, all_outputs[:, 2]).tolist()
        memories = all_outputs[:, 3:3 + NUM_MEMORY].tolist()
        for c, sig, mem, x, y in zip(all_creatures, signals, memories, all_inputs, all_outputs):
            c.signal, c.memory = sig, mem
            c.brain_inputs, c.brain_outputs = x, y
    tick = sim_state["tick"]
    for c, raw in zip(all_creatures, all_outputs):
        outputs = _apply_mate_seeking_drive(c, list(raw), herb_grid, carn_grid)
        c.mate_override = outputs[0] != raw[0] or outputs[1] != raw[1]
        c.act(outputs, sim_state["food_items"], sim_state["creatures"], sim_state["obstacles"], grid,
              food_grid=food_grid, herb_grid=herb_grid, carcass_grid=carcass_grid, tick=tick,
              lakes=lakes)

    alive_herbs, alive_carns = [], []
    for group, alive in ((sim_state["creatures"], alive_herbs),
                         (sim_state["carnivores"], alive_carns)):
        for c in group:
            if c.is_alive():
                alive.append(c)
            else:
                carcasses.append(Carcass(c))
                _retire_creature(c, sim_state, prometheus_metrics, worker_id)
    sim_state["creatures"] = alive_herbs
    sim_state["carnivores"] = alive_carns
    autosave_hall_of_fame(sim_state, prometheus_metrics, worker_id)

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

    if ELITE_REINJECT_INTERVAL and sim_state["tick"] % ELITE_REINJECT_INTERVAL == 0:
        _reinject_elites(sim_state)

    if sim_state["tick"] % SCORE_SAMPLE_INTERVAL == 0:
        sim_state["herbivore_score_history"].append(sim_state["hall_of_fame"]["herbivore"]["score"])
        sim_state["carnivore_score_history"].append(sim_state["hall_of_fame"]["carnivore"]["score"])
        sim_state["herbivore_longevity_history"].append(sim_state["max_age_herbivore"])
        sim_state["carnivore_longevity_history"].append(sim_state["max_age_carnivore"])
        sample_genes(sim_state)

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
        # Eligibility only changes for creatures that reproduce this tick, and those are
        # excluded via reproduced_ids, so it can be computed once up front.
        eligible = [
            p for p in group
            if p.is_adult
            and p.health > max_health * REPRODUCTION_HEALTH_THRESHOLD
            and p.reproductions < MAX_REPRODUCTIONS
            and (p.last_reproduction_tick is None
                 or tick - p.last_reproduction_tick > REPRODUCTION_COOLDOWN_TICKS)
        ]
        if len(eligible) < 2:
            continue
        xy = np.array([(p.pos.x, p.pos.y) for p in eligible])
        close = np.hypot(xy[:, None, 0] - xy[None, :, 0],
                         xy[:, None, 1] - xy[None, :, 1]) < MATE_SELECTION_RADIUS
        np.fill_diagonal(close, False)
        for i, p1 in enumerate(eligible):
            if p1.id in reproduced_ids or len(group) + len(new_offspring) >= pop_cap:
                continue
            constraints = CARNIVORE_GENE_MIN_MAX if p1.is_carnivore else GENE_MIN_MAX
            potential_mates = [eligible[j] for j in np.flatnonzero(close[i])
                               if eligible[j].id not in reproduced_ids
                               and compatible(p1.genes, eligible[j].genes, constraints)]
            if potential_mates:
                best_mate = max(potential_mates,
                                key=_mate_preference(potential_mates, default_genes))
                litter = min(_stochastic_round((p1.genes["litter_size"] + best_mate.genes["litter_size"]) / 2),
                             pop_cap - len(group) - len(new_offspring))
                for _ in range(litter):
                    child_genes = combine_genes(p1.genes, best_mate.genes, default_genes,
                                                gene_constraints=constraints)
                    child_brain = combine_brains(p1.brain, best_mate.brain)
                    mutate_brain(child_brain, child_genes)
                    child = Creature(
                        sim_state["world_bounds"], brain=child_brain,
                        is_carnivore=p1.is_carnivore, genes=child_genes,
                        vec2_factory=p1._vec2, rect_factory=p1._rect,
                    )
                    child.pos = (p1.pos + best_mate.pos) / 2
                    if litter > 1:
                        a = random.uniform(0, 2 * math.pi)
                        child.pos += child._vec2(math.cos(a), math.sin(a)) * LITTER_SPREAD
                    child.health = child.max_health * min(1.0, LITTER_ENDOWMENT / litter)
                    child.lineage = p1.lineage
                    child.parent_ids = (p1.id, best_mate.id)
                    child.descent = max(p1.descent, best_mate.descent) + 1
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


def create_initial_population(count, is_carnivore, world_bounds, brain_file=None,
                               *, vec2_factory=None, rect_factory=None, save_dir="saved_brains"):
    default_genes = DEFAULT_CARNIVORE_GENES if is_carnivore else DEFAULT_HERBIVORE_GENES
    species_name = "carnivore" if is_carnivore else "herbivore"
    constraints = CARNIVORE_GENE_MIN_MAX if is_carnivore else GENE_MIN_MAX
    kwargs = {"vec2_factory": vec2_factory, "rect_factory": rect_factory}
    try:
        if brain_file is None:
            raise FileNotFoundError
        progenitor_brain, gen, progenitor_genes = NeuralNetwork.load(brain_file, BRAIN_TOPOLOGY,
                                                                     save_dir=save_dir)
        if progenitor_genes is None:
            progenitor_genes = default_genes
        print(f"Loading initial {species_name} population from {brain_file} (Gen {gen}).")
        new_pop = [Creature(world_bounds, brain=progenitor_brain, is_carnivore=is_carnivore,
                            genes=progenitor_genes, **kwargs)]
        for _ in range(count - 1):
            mutated_brain = progenitor_brain.copy()
            mutated_genes = mutate_genes(progenitor_genes, default_genes, gene_constraints=constraints)
            mutate_brain(mutated_brain, mutated_genes)
            new_pop.append(Creature(world_bounds, brain=mutated_brain, is_carnivore=is_carnivore,
                                    genes=mutated_genes, **kwargs))
        return new_pop, gen
    except FileNotFoundError:
        print(f"No saved brain found for {species_name}s. Creating a new random population.")
        return [Creature(world_bounds, is_carnivore=is_carnivore, **kwargs) for _ in range(count)], 1


def create_start_populations(world_bounds, source, *, vec2_factory=None, rect_factory=None):
    """(herbivores, carnivores, generation) from the best_brain_*.json in folder source, or
    random populations when source is None."""
    pops, gens = [], []
    for count, is_carnivore, species in ((NUM_HERBIVOROUS, False, "herbivore"),
                                         (NUM_CARNIVORES, True, "carnivore")):
        pop, gen = create_initial_population(
            count, is_carnivore, world_bounds,
            f"best_brain_{species}.json" if source else None,
            vec2_factory=vec2_factory, rect_factory=rect_factory,
            save_dir=source or "saved_brains")
        pops.append(pop)
        gens.append(gen)
    return pops[0], pops[1], max(gens)


def restore_start_progress(sim_state, source):
    """restore_progress for each species loaded from source; population[0] is the unmutated
    progenitor create_initial_population puts first."""
    if source is None:
        return
    for species, key in (("herbivore", "creatures"), ("carnivore", "carnivores")):
        filename = f"best_brain_{species}.json"
        if sim_state[key] and read_saved_score(filename, save_dir=source) is not None:
            progenitor = sim_state[key][0]
            restore_progress(sim_state, species, progenitor.brain, progenitor.genes, filename,
                             save_dir=source)


def _fmt_genes(genes):
    if not genes:
        return "n/a"
    g = genes
    return (
        f"spd:{g.get('max_speed', 0):.2f}  sz:{g.get('size', 0):.1f}"
        f"  sight:{g.get('sight_distance', 0):.0f}  stam:{g.get('max_stamina', 0):.0f}"
    )
