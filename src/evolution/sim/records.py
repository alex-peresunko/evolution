"""Hall of Fame and elite pool: the best brains of retired creatures, per species."""
from dataclasses import dataclass, replace

from ..config import ELITE_POOL_SIZE
from ..species import ALL_SPECIES


@dataclass
class Record:
    score: float
    brain: object = None
    genes: dict | None = None
    generation: int = 1
    fresh: bool = False          # set during this run, so the exit save writes it
    source: str | None = None    # where it came from, for the save log


@dataclass
class SavedScore:
    """Score of the best brain as last written to disk, and that score faded since."""

    raw: float
    faded: float


class SpeciesRecords:
    def __init__(self):
        self.best = Record(score=-1)
        self.elites: list[Record] = []
        self.saved: SavedScore | None = None

    @property
    def has_brain(self):
        return self.best.brain is not None

    def retire(self, c):
        """Enter a dead creature into the elite pool; returns whether it set a new best."""
        record = Record(c.score, c.brain, c.genes, c.generation)
        self._add_elite(record)
        if c.score <= self.best.score:
            return False
        self.best = replace(record, brain=c.brain.copy(), genes=c.genes.copy(), fresh=True,
                            source=f"creature #{c.id}, age {c.age}, lineage #{c.lineage}")
        return True

    def _add_elite(self, record):
        """Keep the ELITE_POOL_SIZE highest-scoring records."""
        pool = self.elites
        if len(pool) < ELITE_POOL_SIZE:
            pool.append(record)
            return
        worst = min(range(len(pool)), key=lambda i: pool[i].score)
        if record.score > pool[worst].score:
            pool[worst] = record

    def seed(self, record):
        """Continue from a loaded brain: it is the best and the only elite, saved as is."""
        self.best = record
        self.elites = [replace(record)]
        self.saved = SavedScore(record.score, record.score)

    def fade(self, factor):
        for record in [self.best, *self.elites]:
            if record.score > 0:
                record.score *= factor
        if self.saved is not None and self.saved.faded > 0:
            self.saved.faded *= factor


class HallOfFame:
    """SpeciesRecords per species; hall_of_fame[species].best is that species' best brain."""

    def __init__(self):
        self._records = {s: SpeciesRecords() for s in ALL_SPECIES}

    def __getitem__(self, species):
        return self._records[species]

    def fade(self, factor):
        for records in self._records.values():
            records.fade(factor)
