"""Pretrain both species' brains by imitating evolution.brain.imitation.teacher_actions and save them
as a history snapshot (game.py --load), so evolution starts from creatures that already seek food
and flee.

Data collection is DAgger-style: round 1 drives the world with random brains, later rounds with
the current student, so the teacher labels the states the student actually visits."""
import argparse
import random

import numpy as np

from evolution import config
from evolution.app.bootstrap import WORLD_BOUNDS
from evolution.brain.imitation import loss, teacher_actions, train_brain
from evolution.brain.network import NeuralNetwork
from evolution.persistence.brain_history import new_snapshot_dir, prune
from evolution.persistence.brain_store import save_brain
from evolution.sim.populations import founders
from evolution.sim.simulation import Simulation
from evolution.species import ALL_SPECIES
from evolution.world.food import Food
from evolution.world.terrain import Obstacle
from evolution.world.world import World
from evolution.world.worldgen import generate_lakes


def build_simulation():
    """A world with random brains and plain rock obstacles; nothing listens for Hall of Fame
    records, so no brain file is written from here."""
    populations = {sp: founders(sp, WORLD_BOUNDS) for sp in ALL_SPECIES}
    food = [Food(WORLD_BOUNDS) for _ in range(config.NUM_FOOD)]
    obstacles = [
        Obstacle(random.randint(100, config.WORLD_WIDTH - 100),
                 random.randint(100, config.WORLD_HEIGHT - 100),
                 random.randint(50, 150), random.randint(50, 150))
        for _ in range(config.NUM_OBSTACLES)
    ]
    return Simulation(World(WORLD_BOUNDS, populations, food, obstacles,
                            generate_lakes(WORLD_BOUNDS, obstacles)))


def collect(students, ticks):
    """Perception rows per species from a fresh world driven by students (None = random)."""
    sim = build_simulation()
    rows = {sp: [] for sp in ALL_SPECIES}
    driven = set()
    for _ in range(ticks):
        for c in sim.world.creatures():
            student = students[c.species]
            if student is not None and c.id not in driven:
                c.brain = student.copy()
                driven.add(c.id)
        sim.step()
        for c in sim.world.creatures():
            if c.brain_inputs is not None:
                rows[c.species].append(c.brain_inputs)
    return {sp: np.array(r) for sp, r in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=1500, help="world ticks per round")
    parser.add_argument("--steps", type=int, default=4000, help="Adam steps per round")
    parser.add_argument("--install", action="store_true",
                        help="overwrite best_brain_*.json instead of adding a history snapshot")
    args = parser.parse_args()

    students = {sp: None for sp in ALL_SPECIES}
    data = {sp: [] for sp in ALL_SPECIES}
    for r in range(1, args.rounds + 1):
        rows = collect(students, args.ticks)
        for sp in ALL_SPECIES:
            data[sp].append(rows[sp])
            X = np.concatenate(data[sp])
            Y = teacher_actions(X, np.full(len(X), sp.is_predator))
            brain = students[sp].copy() if students[sp] else NeuralNetwork(config.BRAIN_TOPOLOGY)
            before = loss(brain, X, Y)
            after = train_brain(brain, X, Y, steps=args.steps)
            students[sp] = brain
            print(f"round {r} {sp.name}: {len(X)} rows, loss {before:.4f} -> {after:.4f}")

    save_dir = "saved_brains" if args.install else new_snapshot_dir("pretrained")
    for sp in ALL_SPECIES:
        save_brain(students[sp], sp.brain_file, generation=1, genes=dict(sp.default_genes),
                   score=0.0, save_dir=save_dir)
    if not args.install:
        prune(protect=save_dir)
        print(f"Saved to {save_dir}; start from it with: python game.py --load 1")


if __name__ == "__main__":
    main()
