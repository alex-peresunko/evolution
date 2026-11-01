"""Island workers for background mode: game_headless.py subprocesses that evolve their own
worlds and swap best brains with this one (see persistence.islands)."""
import os
import subprocess
import sys
from multiprocessing import cpu_count

from ..config import ISLAND_IMMIGRANTS, WORKER_SYNC_INTERVAL
from ..persistence.brain_store import SAVE_DIR
from ..persistence.islands import IslandExchange, clear_island_files, worker_file
from ..species import ALL_SPECIES


class WorkerPool:
    def __init__(self, sim, autosaver):
        self.sim = sim
        self.autosaver = autosaver
        self.workers = []        # (id, Popen, log file)
        self.exchange = None

    @property
    def running(self):
        return bool(self.workers)

    def start(self):
        n = max(1, (cpu_count() or 2) - 1)
        os.makedirs(SAVE_DIR, exist_ok=True)
        clear_island_files()
        self.exchange = IslandExchange(self.sim)
        self.exchange.publish()  # workers start from our best
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        for i in range(n):
            log = open(os.path.join(SAVE_DIR, f"worker_{i}.log"), "w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "game_headless.py", "--worker-id", str(i)],
                                    stdout=log, stderr=log, env=env)
            self.workers.append((i, proc, log))
        print(f"[Islands] Started {n} worker process(es). Logs: saved_brains/worker_N.log")

    def stop(self):
        """Stop the workers (they save their Hall of Fame on SIGTERM), then import what they
        saved."""
        if not self.workers:
            return
        for _, proc, _ in self.workers:
            proc.terminate()
        for _, proc, log in self.workers:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
        self.sync()
        self.workers = []
        print("[Islands] Worker processes stopped.")

    def sync_due(self):
        if self.workers and self.sim.tick % WORKER_SYNC_INTERVAL == 0:
            self.sync()

    def sync(self):
        """Import worker brains that beat the main Hall of Fame (immigrants join the
        population), then publish the main Hall of Fame for the workers."""
        for worker_id, _, _ in self.workers:
            for species in ALL_SPECIES:
                imported = self.exchange.import_if_better(species,
                                                          worker_file(worker_id, species))
                if imported:
                    old_score, new_score = imported
                    self.autosaver.mark_unsaved(species)  # persisted by the next autosave
                    print(f"[Island sync] {species.title} HoF upgraded by worker {worker_id}: "
                          f"{old_score:.3f} -> {new_score:.3f} (+{new_score - old_score:.3f}); "
                          f"{ISLAND_IMMIGRANTS} immigrants joined")
        self.exchange.publish()
