"""Island model for background mode: game.py runs worker copies of game_headless.py, and the
main sim and the workers swap their best brains through files in saved_brains/.

Each worker writes its Hall of Fame to worker_{id}_{species}.json; the main sim imports any that
beats its own and publishes its Hall of Fame (the best across all islands) to
worker_main_{species}.json, which every worker imports in turn."""
import glob
import json
import os

from ..brain.network import NeuralNetwork
from ..config import BRAIN_TOPOLOGY, ISLAND_IMMIGRANTS
from ..sim.records import Record
from ..species import ALL_SPECIES
from .brain_store import SAVE_DIR, save_brain


def worker_file(worker_id, species):
    return f"worker_{worker_id}_{species.name}.json"


def main_file(species):
    return f"worker_main_{species.name}.json"


def clear_island_files(save_dir=SAVE_DIR):
    """Remove exchange files left by a previous run (they may come from an older config)."""
    for path in glob.glob(os.path.join(save_dir, "worker_*.json*")):  # incl. stray .tmp files
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


class IslandExchange:
    """One simulation's side of the brain exchange; remembers which files it already read and
    which brains it already published."""

    def __init__(self, sim, save_dir=SAVE_DIR):
        self.sim = sim
        self.save_dir = save_dir
        self._seen_mtimes = {}
        self._published = {}

    def import_if_better(self, species, filename):
        """If the file changed since last seen and its score beats this sim's Hall of Fame,
        adopt it: the Hall of Fame takes the brain and ISLAND_IMMIGRANTS creatures carrying it
        replace the weakest. Returns (old_score, new_score) on import, else None."""
        path = os.path.join(self.save_dir, filename)
        try:
            mtime = os.path.getmtime(path)
            if mtime <= self._seen_mtimes.get(filename, 0):
                return None
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None  # missing, or mid-replace on some platforms: retried next sync
        self._seen_mtimes[filename] = mtime
        score = data.get("score")
        records = self.sim.hall_of_fame[species]
        if score is None or score <= records.best.score or "weights" not in data:
            return None
        old_score = records.best.score
        records.best = Record(score, NeuralNetwork.from_data(data, BRAIN_TOPOLOGY),
                              {**species.default_genes, **(data.get("genes") or {})},
                              data.get("generation", 1), fresh=True,
                              source=f"imported from {filename}")
        self.sim.replace_weakest(species, records.best, ISLAND_IMMIGRANTS)
        return old_score, score

    def publish(self):
        """Write each species' Hall of Fame to its main_file when it changed since last
        published."""
        for species in ALL_SPECIES:
            best = self.sim.hall_of_fame[species].best
            if best.brain is None or best.brain is self._published.get(species):
                continue
            save_brain(best.brain, main_file(species), generation=self.sim.generation,
                       genes=best.genes, score=best.score, save_dir=self.save_dir)
            self._published[species] = best.brain
