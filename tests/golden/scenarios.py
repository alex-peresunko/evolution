"""Deterministic scenarios observed by the golden trace.

This is the only golden module that knows the simulation's API: when the API moves, update the
adapter here, never the observations' meaning (each field must keep producing the same values).
"""
import contextlib
import hashlib
import io
import json
import os
import random
import time
from unittest import mock

import numpy as np

from evolution import config
from evolution import simulation as sim
from evolution.entities import Carcass, Creature, Food
from evolution.imitation import teacher_actions, train_brain
from evolution.islands import import_if_better, main_file, publish_hall_of_fame
from evolution.neural_network import NeuralNetwork

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)
FROZEN_TIME = 1_700_000_000.0


def _reset(seed):
    random.seed(seed)
    np.random.seed(seed)
    Creature._id_counter = 1
    Food._id_counter = -1
    Carcass._id_counter = -10**9
    sim._stacked_weights.update(key=None, layers=None)


def _brain(nn):
    return [w.tolist() for w in nn.weights] + [b.tolist() for b in nn.biases]


def _creature(c):
    return [
        c.id, c.is_carnivore, c.pos.x, c.pos.y, c.angle, c.health, c.max_health, c.stamina,
        c.max_stamina, c.fat, c.hunger, c.score, c.age, c.generation, c.lineage, c.descent,
        list(c.parent_ids), c.reproductions, c.last_reproduction_tick, c.size, c.max_speed,
        c.current_speed, c.signal, list(c.memory), c.eating_ticks, c.meal_health,
        c.meal_carcass.id if c.meal_carcass is not None else None, c.winded, c.in_water,
        c.is_adult, c.is_best, c.respawn_immunity_ticks, c.last_bite_tick, c.last_hit_tick,
        c.genes, list(c.whiskers), list(c.target_whiskers), c.effective_sight,
        c._nearest_target_dist, c._nearest_smell_strength, c.mate_override,
        None if c.brain_inputs is None else list(c.brain_inputs),
        None if c.brain_outputs is None else list(c.brain_outputs),
    ]


def _record(r):
    return {"score": r["score"], "generation": r.get("generation"), "fresh": r.get("fresh"),
            "source": r.get("source"), "genes": r.get("genes"),
            "brain": None if r.get("brain") is None else _brain(r["brain"])}


def observe(state):
    """One observation of the whole simulation as plain data, one entry per field."""
    herbs, carns = state["creatures"], state["carnivores"]
    return {
        "herbivores": [_creature(c) for c in herbs],
        "carnivores": [_creature(c) for c in carns],
        "brains": [_brain(c.brain) for c in herbs + carns],
        "food": [(f.id, f.pos.x, f.pos.y, f.lifetime, f.consumed) for f in state["food_items"]],
        "carcasses": [(cc.id, cc.pos.x, cc.pos.y, cc.radius, cc.is_carnivore, cc.max_meat,
                       cc.meat, cc.age) for cc in state["carcasses"]],
        "patches": [(p["x"], p["y"], p["heading"]) for p in state["food_patches"]],
        "clock": (state["tick"], state["light"], state["season"]),
        "counters": (state["generation"], state["extinctions"], state["max_age_herbivore"],
                     state["max_age_carnivore"], state["last_autosave_tick"],
                     sorted(state["hof_unsaved"])),
        "histories": {sp: (state[f"{sp}_score_history"], state[f"{sp}_longevity_history"])
                      for sp in ("herbivore", "carnivore")},
        "gene_history": {sp: {g: list(v) for g, v in genes.items()}
                         for sp, genes in state["gene_history"].items()},
        "lineages": state["lineages"],
        "hall_of_fame": {sp: _record(r) for sp, r in state["hall_of_fame"].items()},
        "elites": {sp: [_record(r) for r in pool] for sp, pool in state["elites"].items()},
        "saved_scores": state.get("hof_saved_scores", {}),
    }


def terrain(state):
    return {
        "obstacles": [(type(o).__name__, o.rect.x, o.rect.y, o.rect.w, o.rect.h,
                       getattr(o, "x", None), getattr(o, "y", None),
                       getattr(o, "crown_radius", None)) for o in state["obstacles"]],
        "lakes": [(lk.x, lk.y, lk.radius, lk.max_radius, list(lk.harmonics))
                  for lk in state["lakes"]],
    }


def _files(directory):
    out = {}
    for name in sorted(os.listdir(directory)):
        with open(os.path.join(directory, name), "rb") as f:
            out[name] = f.read().decode()
    return out


def new_world(source=None, vec2_factory=None, rect_factory=None):
    herbs, carns, generation = sim.create_start_populations(
        WORLD_BOUNDS, source, vec2_factory=vec2_factory, rect_factory=rect_factory)
    food = [Food(WORLD_BOUNDS, vec2_factory=vec2_factory) for _ in range(config.NUM_FOOD)]
    obstacles = sim.generate_obstacles(WORLD_BOUNDS, rect_factory=rect_factory)
    state = sim.build_sim_state(WORLD_BOUNDS, herbs, carns, food, obstacles,
                                sim.generate_lakes(WORLD_BOUNDS, obstacles))
    state["generation"] = generation
    sim.restore_start_progress(state, source)
    return state


def step(state):
    sim.update_world(state)


def save_on_exit(state):
    sim.save_hall_of_fame(state)


def evolution_run(workdir, seed=1234, ticks=3000, every=50):
    """Random start, long run with autosaves, then the exit save; then a second world started
    from those files with an island exchange. Yields (label, observation) pairs."""
    _reset(seed)
    log = io.StringIO()
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        with contextlib.redirect_stdout(log):
            state = new_world()
        yield "terrain", terrain(state)
        for t in range(1, ticks + 1):
            with contextlib.redirect_stdout(log):
                step(state)
            if t % every == 0:
                yield f"tick {t}", {**observe(state), "log": log.getvalue()}
                log.seek(0)
                log.truncate()
        with contextlib.redirect_stdout(log):
            save_on_exit(state)
        yield "exit save", {"files": _files("saved_brains"), "log": log.getvalue()}

        _reset(seed + 1)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            state = new_world(source="saved_brains")
        yield "reload", {**observe(state), **terrain(state), "log": log.getvalue()}
        for t in range(1, 301):
            if t == 150:  # wipe out the carnivores: carcasses, retirement, repopulation
                for c in state["carnivores"]:
                    c.health = 0
            with contextlib.redirect_stdout(log):
                step(state)
        yield "reload run", {**observe(state), "log": log.getvalue()}

        better = NeuralNetwork(config.BRAIN_TOPOLOGY)
        better.save("worker_0_herbivore.json", save_dir="saved_brains", generation=7,
                    score=1e6, genes={"size": 12.0})
        with contextlib.redirect_stdout(log):
            imported = import_if_better(state, "herbivore", "worker_0_herbivore.json", {})
            published = {}
            publish_hall_of_fame(state, published)
            for _ in range(200):
                step(state)
            sim.autosave_hall_of_fame(state, force=True)
        with open(os.path.join("saved_brains", main_file("carnivore"))) as f:
            main_carn = json.load(f)
        yield "islands", {**observe(state), "imported": imported, "main_carn": main_carn,
                          "files": _files("saved_brains"), "log": log.getvalue()}
    finally:
        os.chdir(cwd)


def imitation_run(seed=99):
    """Teacher labels and a short training run on perception rows of a seeded world."""
    _reset(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        state = new_world()
        for _ in range(60):
            step(state)
    everyone = state["creatures"] + state["carnivores"]
    X = np.array([c.brain_inputs for c in everyone])
    carn = np.array([c.is_carnivore for c in everyone])
    Y = teacher_actions(X, carn)
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    final = train_brain(brain, X, Y, steps=50, batch_size=64, rng=np.random.default_rng(seed))
    return {"X": X, "Y": Y, "loss": final, "brain": _brain(brain)}


def render_run(seed=7, ticks=240, every=40):
    """Frames of the full game.py screen (world, minimap, HUD, brain panel, gene chart) under
    pygame's dummy video driver, at two zoom levels. Yields (label, frame bytes)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    import game

    _reset(seed)
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    font = pygame.font.Font(None, config.HUD_FONT_SIZE)
    clock = mock.Mock(get_fps=lambda: 60.0)
    with mock.patch.object(time, "time", lambda: FROZEN_TIME), \
            contextlib.redirect_stdout(io.StringIO()):
        state = new_world(vec2_factory=game._vec2, rect_factory=game._rect)
        from collections import deque

        from evolution.render import Renderer
        state.update({
            "paused": False, "clock": clock, "selected_creature": None, "fps_limited": True,
            "drawing_enabled": True, "background_mode": False, "worker_processes": [],
            "cam_x": 0.0, "cam_y": 0.0, "zoom": 1.0, "minimap_dragging": False,
            "ticks_per_frame": 1, "chart_scale": config.CHART_SCALE_DEFAULT, "tps": 0,
            "show_brain": True, "show_genes": True,
            "brain_history": deque(maxlen=config.BRAIN_VIEW_HISTORY),
            "brain_history_owner": None,
            "renderer": Renderer(config.WORLD_WIDTH, config.WORLD_HEIGHT, state["obstacles"],
                                 state["lakes"]),
        })
        game.MINIMAP_RECT = pygame.Rect(config.MINIMAP_X, config.MINIMAP_Y,
                                        config.MINIMAP_WIDTH, config.MINIMAP_HEIGHT)
        for t in range(1, ticks + 1):
            step(state)
            if t == 20:
                state["selected_creature"] = state["carnivores"][0]
            game._record_brain(state)
            game._follow_selected(state)
            if t == ticks // 2:
                game._zoom(state, 7, (700, 450))
            if t % every == 0:
                game.draw_elements(screen, font, state)
                pixels = hashlib.sha256(pygame.image.tobytes(screen, "RGB")).hexdigest()
                yield f"frame {t}", {"pixels": pixels,
                                     "sim": observe(state)}
    pygame.quit()
