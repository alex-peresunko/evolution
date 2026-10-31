"""Writing Hall of Fame brains to disk while a simulation runs, and when it exits."""
from ..config import AUTOSAVE_INTERVAL
from ..sim.events import SimulationListener
from ..sim.records import SavedScore
from ..species import ALL_SPECIES
from .brain_store import SAVE_DIR, save_brain, update_saved_history
from .islands import worker_file


def log_prefix(worker_id):
    return f"[W{worker_id}] " if worker_id is not None else ""


class HallOfFameAutosaver(SimulationListener):
    """Saves each new Hall of Fame best at most once per AUTOSAVE_INTERVAL ticks; a record set
    inside the interval waits for a later save instead of being dropped.

    Island workers (worker_id set) save to their exchange files instead of best_brain_*.json."""

    def __init__(self, worker_id=None, uptime=None, save_dir=SAVE_DIR):
        self.worker_id = worker_id
        self.uptime = uptime              # () -> seconds, stored in saved files
        self.save_dir = save_dir
        self.unsaved = set()
        self.last_save_tick = 0

    def filename(self, species):
        if self.worker_id is not None:
            return worker_file(self.worker_id, species)
        return species.brain_file

    def mark_unsaved(self, species):
        self.unsaved.add(species)

    def record_set(self, sim, species):
        self.mark_unsaved(species)
        self.save_due(sim)

    def deaths_processed(self, sim):
        self.save_due(sim)

    def save_due(self, sim, force=False):
        if not self.unsaved or (not force and sim.tick - self.last_save_tick < AUTOSAVE_INTERVAL):
            return
        for species in sorted(self.unsaved, key=lambda s: s.name):
            best = sim.hall_of_fame[species].best
            filename = self.filename(species)
            save_brain(best.brain, filename, generation=sim.generation, genes=best.genes,
                       score=best.score, uptime_seconds=self._uptime(),
                       history=sim.stats[species].saved_history(), save_dir=self.save_dir)
            self._log_save(sim, species, filename)
            self.last_save_tick = sim.tick
        self.unsaved.clear()

    def _uptime(self):
        return self.uptime() if self.uptime is not None else None

    def _log_save(self, sim, species, filename):
        records = sim.hall_of_fame[species]
        previous, score = records.saved, records.best.score
        if previous is None:
            change = f"{score:.3f} (new)"
        else:
            old = f"{previous.raw:.3f}"
            if f"{previous.faded:.3f}" != old:
                old += f" (faded to {previous.faded:.3f})"
            change = f"{old} -> {score:.3f} ({score - previous.faded:+.3f})"
        details = [f"gen {records.best.generation}"]
        if records.best.source:
            details.append(records.best.source)
        print(f"{log_prefix(self.worker_id)}[HoF save] {species.title} best brain saved to "
              f"{filename}: {change}; {', '.join(details)}; tick {sim.tick}")
        records.saved = SavedScore(score, score)


def save_on_exit(sim, uptime=None, save_dir=SAVE_DIR):
    """Each species' Hall of Fame into its best_brain file if it was set during this run
    (records fade, so it may score below the file), otherwise only the file's chart history."""
    for species in ALL_SPECIES:
        best = sim.hall_of_fame[species].best
        history = sim.stats[species].saved_history()
        if best.brain is None:
            print(f"No {species.name} in Hall of Fame to save.")
        elif best.fresh:
            save_brain(best.brain, species.brain_file, generation=sim.generation,
                       genes=best.genes, score=best.score,
                       uptime_seconds=uptime() if uptime is not None else None,
                       history=history, save_dir=save_dir)
        else:
            update_saved_history(species.brain_file, history, save_dir=save_dir)
            print(f"Kept existing {species.brain_file} (no new record this run), "
                  "updated its chart history")
