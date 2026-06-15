import sys
import time
import random

from prometheus_client import start_http_server

from evolution import config
from evolution.entities import Creature, Food, Obstacle
from evolution.simulation import update_world, create_initial_population, build_sim_state
from evolution.metrics import GameMetrics
from evolution.brain_store import save_brain


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
        hof_h = sim_state["hall_of_fame"]["herbivore"]
        if hof_h["brain"] is not None:
            save_brain(hof_h["brain"], "best_brain_herbivore.json",
                       generation=sim_state["generation"], genes=hof_h["genes"],
                       score=hof_h["score"], metrics=prometheus_metrics)
        else:
            print("No herbivore in Hall of Fame to save.")
        hof_c = sim_state["hall_of_fame"]["carnivore"]
        if hof_c["brain"] is not None:
            save_brain(hof_c["brain"], "best_brain_carnivore.json",
                       generation=sim_state["generation"], genes=hof_c["genes"],
                       score=hof_c["score"], metrics=prometheus_metrics)
        else:
            print("No carnivore in Hall of Fame to save.")
        print("Exiting.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=None)
    args = parser.parse_args()
    if args.worker_id is None:
        start_http_server(8000)
    main(worker_id=args.worker_id)
