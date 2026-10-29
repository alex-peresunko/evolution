import sys
import time
import random

from prometheus_client import start_http_server

from evolution import config
from evolution.entities import Creature, Food, Obstacle
from evolution.simulation import (update_world, create_initial_population, build_sim_state,
                                  restore_progress, species_history)
from evolution.metrics import GameMetrics
from evolution.brain_store import read_saved_score, save_brain_if_better, update_saved_history


def main(worker_id=None):
    world_bounds = (config.WORLD_WIDTH, config.WORLD_HEIGHT)
    herbivores, h_gen = create_initial_population(
        config.NUM_HERBIVOROUS, False, world_bounds, "best_brain_herbivore.json"
    )
    carnivores, c_gen = create_initial_population(
        config.NUM_CARNIVORES, True, world_bounds, "best_brain_carnivore.json"
    )

    food_items = [Food(world_bounds) for _ in range(config.NUM_FOOD)]
    obstacles = [
        Obstacle(
            random.randint(100, config.WORLD_WIDTH - 100),
            random.randint(100, config.WORLD_HEIGHT - 100),
            random.randint(50, 150),
            random.randint(50, 150),
        )
        for _ in range(config.NUM_OBSTACLES)
    ]

    sim_state = build_sim_state(world_bounds, herbivores, carnivores, food_items, obstacles)
    sim_state["generation"] = max(h_gen, c_gen)
    for species, population in (("herbivore", herbivores), ("carnivore", carnivores)):
        filename = f"best_brain_{species}.json"
        if read_saved_score(filename) is not None:  # population[0] is the unmutated progenitor
            restore_progress(sim_state, species, population[0].brain, population[0].genes, filename)
    prometheus_metrics = GameMetrics(sim_state)

    last_time = time.time()
    ticks_this_second = 0
    last_tps_report_time = time.time()

    print(f"Headless simulation starting at Generation {sim_state['generation']}... Press Ctrl+C to stop.")
    try:
        while sim_state["running"]:
            now = time.time()
            dt = now - last_time
            last_time = now

            update_world(sim_state, prometheus_metrics, worker_id=worker_id)
            prometheus_metrics.update(sim_state, dt)

            ticks_this_second += 1
            if now - last_tps_report_time >= 1.0:
                tps = ticks_this_second
                ticks_this_second = 0
                last_tps_report_time = now
                hof_h = sim_state["hall_of_fame"]["herbivore"]["score"]
                hof_c = sim_state["hall_of_fame"]["carnivore"]["score"]
                pfx = f"[W{worker_id}] " if worker_id is not None else ""
                sys.stdout.write(
                    f"\r{pfx}Gen {sim_state['generation']} tick={sim_state['tick']:7d}"
                    f" | TPS:{tps:4d}"
                    f" | H:{len(sim_state['creatures']):3d} C:{len(sim_state['carnivores']):3d}"
                    f" | HoF H:{hof_h:.3f} C:{hof_c:.3f}"
                    f" | LonH:{sim_state['max_age_herbivore']} LonC:{sim_state['max_age_carnivore']}   "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        print("\nSaving best brains from Hall of Fame on exit...")
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
        print("Exiting.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=None)
    args = parser.parse_args()
    if args.worker_id is None:
        start_http_server(8000)
    main(worker_id=args.worker_id)
