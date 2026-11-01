"""Deterministic scenarios observed by the golden trace.

This is the only golden module that knows the simulation's API: when the API moves, update the
adapter here, never the observations' meaning (each field must keep producing the same values).
"""
import contextlib
import functools
import hashlib
import io
import json
import os
import random
import time
from unittest import mock

import numpy as np

from evolution import config
from evolution.app import bootstrap
from evolution.app.console import ExtinctionReporter
from evolution.brain.imitation import teacher_actions, train_brain
from evolution.brain.network import NeuralNetwork
from evolution.persistence.autosave import HallOfFameAutosaver
from evolution.persistence.autosave import save_on_exit as _save_on_exit
from evolution.persistence.brain_store import save_brain
from evolution.persistence.islands import IslandExchange, main_file
from evolution.species import ALL_SPECIES, CARNIVORE, HERBIVORE
from evolution.world.creature import Creature
from evolution.world.food import Carcass, Food
from evolution.world.geometry import HEADLESS

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)
FROZEN_TIME = 1_700_000_000.0


def _reset(seed):
    random.seed(seed)
    np.random.seed(seed)
    Creature._id_counter = 1
    Food._id_counter = -1
    Carcass._id_counter = -10**9


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
        c.nearest_target_dist, c.nearest_smell_strength, c.mate_override,
        None if c.brain_inputs is None else list(c.brain_inputs),
        None if c.brain_outputs is None else list(c.brain_outputs),
    ]


def _record(r):
    return {"score": r.score, "generation": r.generation if r.brain is not None else None,
            "fresh": r.fresh or None, "source": r.source, "genes": r.genes,
            "brain": None if r.brain is None else _brain(r.brain)}


def _autosaver(sim):
    return next(x for x in sim.listeners if isinstance(x, HallOfFameAutosaver))


def observe(sim):
    """One observation of the whole simulation as plain data, one entry per field."""
    world, hof, stats = sim.world, sim.hall_of_fame, sim.stats
    herbs, carns = world.herbivores, world.carnivores
    autosaver = _autosaver(sim)
    return {
        "herbivores": [_creature(c) for c in herbs],
        "carnivores": [_creature(c) for c in carns],
        "brains": [_brain(c.brain) for c in herbs + carns],
        "food": [(f.id, f.pos.x, f.pos.y, f.lifetime, f.consumed) for f in world.food],
        "carcasses": [(cc.id, cc.pos.x, cc.pos.y, cc.radius, cc.species.is_predator,
                       cc.max_meat, cc.meat, cc.age) for cc in world.carcasses],
        "patches": [(p.x, p.y, p.heading) for p in world.food_patches],
        "clock": (world.tick, world.light, world.season),
        "counters": (sim.generation, sim.extinctions, stats[HERBIVORE].max_age,
                     stats[CARNIVORE].max_age, autosaver.last_save_tick,
                     sorted(s.name for s in autosaver.unsaved)),
        "histories": {sp.name: (stats[sp].score_history, stats[sp].longevity_history)
                      for sp in ALL_SPECIES},
        "gene_history": {sp.name: {g: list(v) for g, v in stats[sp].gene_history.items()}
                         for sp in ALL_SPECIES},
        "lineages": {sp.name: stats[sp].lineages for sp in ALL_SPECIES},
        "hall_of_fame": {sp.name: _record(hof[sp].best) for sp in ALL_SPECIES},
        "elites": {sp.name: [_record(r) for r in hof[sp].elites] for sp in ALL_SPECIES},
        "saved_scores": {sp.name: {"raw": hof[sp].saved.raw, "faded": hof[sp].saved.faded}
                         for sp in ALL_SPECIES if hof[sp].saved is not None},
    }


def terrain(sim):
    world = sim.world
    return {
        "obstacles": [(type(o).__name__, o.rect.x, o.rect.y, o.rect.w, o.rect.h,
                       getattr(o, "x", None), getattr(o, "y", None),
                       getattr(o, "crown_radius", None)) for o in world.obstacles],
        "lakes": [(lk.x, lk.y, lk.radius, lk.max_radius, list(lk.harmonics))
                  for lk in world.lakes],
    }


def _files(directory):
    out = {}
    for name in sorted(os.listdir(directory)):
        with open(os.path.join(directory, name), "rb") as f:
            out[name] = f.read().decode()
    return out


def new_world(source=None, geometry=HEADLESS):
    return bootstrap.start_simulation(
        source, geometry=geometry, listeners=[HallOfFameAutosaver(), ExtinctionReporter()])


def step(sim):
    sim.step()


def save_on_exit(sim):
    _save_on_exit(sim)


def _plain(value):
    """value with numpy scalars as Python scalars: the fingerprint hashes the repr, and whether
    a value is np.float64 or float is not part of the simulation's behaviour."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(v) for v in value)
    return value


def _plain_observations(run):
    @functools.wraps(run)
    def wrapper(*args, **kwargs):
        for label, observation in run(*args, **kwargs):
            yield label, _plain(observation)
    return wrapper


@_plain_observations
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
                for c in state.world.carnivores:
                    c.health = 0
            with contextlib.redirect_stdout(log):
                step(state)
        yield "reload run", {**observe(state), "log": log.getvalue()}

        better = NeuralNetwork(config.BRAIN_TOPOLOGY)
        save_brain(better, "worker_0_herbivore.json", save_dir="saved_brains", generation=7,
                   score=1e6, genes={"size": 12.0})
        with contextlib.redirect_stdout(log):
            exchange = IslandExchange(state)
            imported = exchange.import_if_better(HERBIVORE, "worker_0_herbivore.json")
            exchange.publish()
            for _ in range(200):
                step(state)
            _autosaver(state).save_due(state, force=True)
        with open(os.path.join("saved_brains", main_file(CARNIVORE))) as f:
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
    everyone = state.world.creatures()
    X = np.array([c.brain_inputs for c in everyone])
    carn = np.array([c.is_carnivore for c in everyone])
    Y = teacher_actions(X, carn)
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    final = train_brain(brain, X, Y, steps=50, batch_size=64, rng=np.random.default_rng(seed))
    return {"X": X, "Y": Y, "loss": final, "brain": _brain(brain)}


@_plain_observations
def render_run(seed=7, ticks=240, every=40):
    """Frames of the full game.py screen (world, minimap, HUD, brain panel, gene chart) under
    pygame's dummy video driver, at two zoom levels. Yields (label, frame bytes)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from evolution.ui.app import PYGAME_GEOMETRY, draw_frame
    from evolution.ui.render import WorldRenderer
    from evolution.ui.view_state import ViewState

    _reset(seed)
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    font = pygame.font.Font(None, config.HUD_FONT_SIZE)
    with mock.patch.object(time, "time", lambda: FROZEN_TIME), \
            contextlib.redirect_stdout(io.StringIO()):
        sim = new_world(geometry=PYGAME_GEOMETRY)
        view = ViewState()
        renderer = WorldRenderer(config.WORLD_WIDTH, config.WORLD_HEIGHT, sim.world.obstacles,
                                 sim.world.lakes)
        for t in range(1, ticks + 1):
            step(sim)
            if t == 20:
                view.selected = sim.world.carnivores[0]
            view.record_brain()
            view.follow_selected(sim.world)
            if t == ticks // 2:
                view.zoom_by(7, (700, 450))
            if t % every == 0:
                draw_frame(screen, font, sim, view, renderer, 60.0)
                pixels = hashlib.sha256(pygame.image.tobytes(screen, "RGB")).hexdigest()
                yield f"frame {t}", {"pixels": pixels,
                                     "sim": observe(sim)}
    pygame.quit()
