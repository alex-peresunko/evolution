"""The headless run loop: game_headless.py, the container, and game.py's island workers."""
import os
import signal
import sys
import time

from ..config import WORKER_SYNC_INTERVAL
from ..persistence.autosave import HallOfFameAutosaver, log_prefix, save_on_exit
from ..persistence.brain_store import SAVE_DIR
from ..persistence.islands import IslandExchange, main_file
from ..species import ALL_SPECIES, CARNIVORE, HERBIVORE
from .bootstrap import archive_previous_best, start_simulation
from .console import ExtinctionReporter


def status_line(sim, tps, prefix=""):
    world, hof, stats = sim.world, sim.hall_of_fame, sim.stats
    return (f"\r{prefix}Gen {sim.generation} Ext {sim.extinctions}"
            f" tick={sim.tick:7d}"
            f" | TPS:{tps:4d}"
            f" | H:{len(world.herbivores):3d} C:{len(world.carnivores):3d}"
            f" | HoF H:{hof[HERBIVORE].best.score:.3f} C:{hof[CARNIVORE].best.score:.3f}"
            f" | LonH:{stats[HERBIVORE].max_age} LonC:{stats[CARNIVORE].max_age}   ")


class HeadlessRunner:
    """Island workers (worker_id set) start from the main sim's best brains, never touch the
    brain history, save to their exchange files and import the main sim's published brains."""

    def __init__(self, source, *, new=False, worker_id=None, metrics=None):
        self.worker_id = worker_id
        self.metrics = metrics
        self.prefix = log_prefix(worker_id)
        uptime = metrics.uptime_seconds if metrics is not None else None
        self.autosaver = HallOfFameAutosaver(worker_id, uptime)
        self.sim = start_simulation(source, listeners=[self.autosaver,
                                                       ExtinctionReporter(worker_id)])
        if metrics is not None:
            metrics.watch(self.sim, self.autosaver)
        if worker_id is None:
            archive_previous_best(source, new)
        self.exchange = IslandExchange(self.sim) if worker_id is not None else None

    def run(self):
        # SIGTERM (game.py stopping its workers, `podman stop`) runs the finally-block save too.
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        parent_pid = os.getppid()
        sim = self.sim
        last_report = time.time()
        ticks_this_second = 0
        print(f"Headless simulation starting at Generation {sim.generation}... "
              "Press Ctrl+C to stop.")
        try:
            while True:
                now = time.time()
                sim.step()
                if self.exchange is not None and sim.tick % WORKER_SYNC_INTERVAL == 0:
                    if os.getppid() != parent_pid:
                        print(f"\n{self.prefix}game.py is gone, exiting.")
                        break
                    self._import_main_brains()
                ticks_this_second += 1
                if now - last_report >= 1.0:
                    sys.stdout.write(status_line(sim, ticks_this_second, self.prefix))
                    sys.stdout.flush()
                    ticks_this_second, last_report = 0, now
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")
        finally:
            self._save()
            print("Exiting.")

    def _import_main_brains(self):
        for species in ALL_SPECIES:  # the best brain across all islands, via game.py
            if self.exchange.import_if_better(species, main_file(species)):
                print(f"\n{self.prefix}Imported main {species.name} brain "
                      f"(score {self.sim.hall_of_fame[species].best.score:.3f})")

    def _save(self):
        if self.worker_id is not None:  # game.py imports these and owns the best_brain files
            print("\nSaving unsaved Hall of Fame records to worker files...")
            self.autosaver.save_due(self.sim, force=True)
        else:
            print("\nSaving best brains from Hall of Fame on exit...")
            save_on_exit(self.sim, uptime=self.autosaver.uptime, save_dir=SAVE_DIR)
