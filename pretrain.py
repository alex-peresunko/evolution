"""Pretrain both species' brains by imitating evolution.imitation.teacher_actions, then save them
so game.py (L) and game_headless.py start from creatures that already seek food and flee.

Data collection is DAgger-style: round 1 drives the world with random brains, later rounds with
the current student, so the teacher labels the states the student actually visits."""
import argparse
import random

import numpy as np

from evolution import config
from evolution.brain_store import save_brain
from evolution.entities import Food, Obstacle
from evolution.imitation import loss, teacher_actions, train_brain
from evolution.neural_network import NeuralNetwork
from evolution.simulation import build_sim_state, create_initial_population, update_world

SPECIES = ("herbivore", "carnivore")


def build_world():
    world_bounds = (config.WORLD_WIDTH, config.WORLD_HEIGHT)
    herbivores, _ = create_initial_population(config.NUM_HERBIVOROUS, False, world_bounds)
    carnivores, _ = create_initial_population(config.NUM_CARNIVORES, True, world_bounds)
    food = [Food(world_bounds) for _ in range(config.NUM_FOOD)]
    obstacles = [
        Obstacle(random.randint(100, config.WORLD_WIDTH - 100),
                 random.randint(100, config.WORLD_HEIGHT - 100),
                 random.randint(50, 150), random.randint(50, 150))
        for _ in range(config.NUM_OBSTACLES)
    ]
    sim_state = build_sim_state(world_bounds, herbivores, carnivores, food, obstacles)
    sim_state["last_autosave_tick"] = float("inf")  # never write best_brain_*.json from here
    return sim_state


def collect(students, ticks):
    """Perception rows per species from a fresh world driven by students (None = random)."""
    sim_state = build_world()
    rows = {sp: [] for sp in SPECIES}
    for _ in range(ticks):
        for c in sim_state["creatures"] + sim_state["carnivores"]:
            student = students["carnivore" if c.is_carnivore else "herbivore"]
            if student is not None and not getattr(c, "_student", False):
                c.brain, c._student = student.copy(), True
        update_world(sim_state)
        for c in sim_state["creatures"] + sim_state["carnivores"]:
            if c.brain_inputs is not None:
                rows["carnivore" if c.is_carnivore else "herbivore"].append(c.brain_inputs)
    return {sp: np.array(r) for sp, r in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=1500, help="world ticks per round")
    parser.add_argument("--steps", type=int, default=4000, help="Adam steps per round")
    parser.add_argument("--install", action="store_true",
                        help="overwrite best_brain_*.json instead of writing pretrained_*.json")
    args = parser.parse_args()

    students = {sp: None for sp in SPECIES}
    data = {sp: [] for sp in SPECIES}
    for r in range(1, args.rounds + 1):
        rows = collect(students, args.ticks)
        for sp in SPECIES:
            data[sp].append(rows[sp])
            X = np.concatenate(data[sp])
            Y = teacher_actions(X, np.full(len(X), sp == "carnivore"))
            brain = students[sp].copy() if students[sp] else NeuralNetwork(config.BRAIN_TOPOLOGY)
            before = loss(brain, X, Y)
            after = train_brain(brain, X, Y, steps=args.steps)
            students[sp] = brain
            print(f"round {r} {sp}: {len(X)} rows, loss {before:.4f} -> {after:.4f}")

    prefix = "best_brain" if args.install else "pretrained_brain"
    for sp in SPECIES:
        default = config.DEFAULT_CARNIVORE_GENES if sp == "carnivore" else \
            config.DEFAULT_HERBIVORE_GENES
        save_brain(students[sp], f"{prefix}_{sp}.json", generation=1, genes=dict(default),
                   score=0.0)


if __name__ == "__main__":
    main()
