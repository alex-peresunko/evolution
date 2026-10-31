import os
import signal
import sys
import time

from prometheus_client import start_http_server

from evolution import config
from evolution.entities import Creature, Food
from evolution.simulation import (update_world, create_start_populations, build_sim_state,
                                  generate_lakes, generate_obstacles,
                                  restore_start_progress, autosave_hall_of_fame,
                                  save_hall_of_fame)
from evolution.metrics import GameMetrics
from evolution.brain_history import (SAVE_DIR, add_start_arguments, archive_after_start,
                                     check_load_argument, format_listing, start_source)
from evolution.islands import SPECIES, import_if_better, main_file


def main(worker_id=None, new=False, load=None):
    world_bounds = (config.WORLD_WIDTH, config.WORLD_HEIGHT)
    # Island workers always start from the main sim's best brains and never touch the history.
    source = SAVE_DIR if worker_id is not None else start_source(new, load)
    herbivores, carnivores, generation = create_start_populations(world_bounds, source)
    food_items = [Food(world_bounds) for _ in range(config.NUM_FOOD)]
    obstacles = generate_obstacles(world_bounds)

    sim_state = build_sim_state(world_bounds, herbivores, carnivores, food_items, obstacles,
                                generate_lakes(world_bounds, obstacles))
    sim_state["generation"] = generation
    restore_start_progress(sim_state, source)
    if worker_id is None:
        snapshot = archive_after_start(source, new)
        if snapshot:
            print(f"Previous best brains kept in {snapshot}")
    prometheus_metrics = GameMetrics(sim_state)
    # SIGTERM (game.py stopping its workers, `docker stop`) runs the finally-block save too.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    parent_pid = os.getppid()
    island_seen = {}

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
            if worker_id is not None and sim_state["tick"] % config.WORKER_SYNC_INTERVAL == 0:
                if os.getppid() != parent_pid:
                    print(f"\n[W{worker_id}] game.py is gone, exiting.")
                    break
                for species in SPECIES:  # the best brain across all islands, via game.py
                    if import_if_better(sim_state, species, main_file(species), island_seen):
                        print(f"\n[W{worker_id}] Imported main {species} brain "
                              f"(score {sim_state['hall_of_fame'][species]['score']:.3f})")

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
        if worker_id is not None:  # game.py imports these and owns the best_brain files
            print("\nSaving unsaved Hall of Fame records to worker files...")
            autosave_hall_of_fame(sim_state, prometheus_metrics, worker_id, force=True)
        else:
            print("\nSaving best brains from Hall of Fame on exit...")
            save_hall_of_fame(sim_state, prometheus_metrics)
        print("Exiting.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=None)
    add_start_arguments(parser)
    args = parser.parse_args()
    if args.list_brains:
        print(format_listing())
        sys.exit(0)
    check_load_argument(parser, args)
    if args.worker_id is None:
        start_http_server(8000)
    main(worker_id=args.worker_id, new=args.new, load=args.load)
