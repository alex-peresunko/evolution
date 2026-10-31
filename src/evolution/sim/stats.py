"""Chart data: Hall of Fame score, longevity, mean genes and lineages over time, per species."""
from collections import deque

from ..config import CHART_HISTORY_SAVE_MAX
from ..species import ALL_SPECIES


class SpeciesStats:
    def __init__(self):
        self.score_history = []
        self.longevity_history = []
        self.max_age = 0
        self.gene_history = {}      # gene -> deque of species means, created at the first sample
        self.lineages = 0

    def note_death(self, c):
        self.max_age = max(self.max_age, c.age)

    def sample(self, best_score):
        self.score_history.append(best_score)
        self.longevity_history.append(self.max_age)

    def sample_genes(self, population, genes):
        for gene in genes:
            series = self.gene_history.setdefault(gene, deque(maxlen=CHART_HISTORY_SAVE_MAX))
            if population:
                series.append(sum(c.genes[gene] for c in population) / len(population))
        self.lineages = len({c.lineage for c in population})

    def saved_history(self):
        """The chart histories, trimmed for saving into the species' brain file."""
        return {"score": list(self.score_history[-CHART_HISTORY_SAVE_MAX:]),
                "longevity": list(self.longevity_history[-CHART_HISTORY_SAVE_MAX:])}

    def restore(self, history):
        self.score_history = list(history.get("score", []))
        self.longevity_history = list(history.get("longevity", []))
        if self.longevity_history:
            self.max_age = max(self.max_age, int(self.longevity_history[-1]))


class Statistics:
    def __init__(self):
        self._stats = {s: SpeciesStats() for s in ALL_SPECIES}

    def __getitem__(self, species):
        return self._stats[species]

    def sample(self, world, hall_of_fame):
        for species, stats in self._stats.items():
            stats.sample(hall_of_fame[species].best.score)
        for species, stats in self._stats.items():
            stats.sample_genes(world.population(species), species.default_genes)

    def restore(self, species, history):
        """Continue a saved species' charts; every series is then left-padded to the longest,
        because all charts share one x axis."""
        self._stats[species].restore(history)
        series = [(s, attr) for s in self._stats.values()
                  for attr in ("score_history", "longevity_history")]
        length = max(len(getattr(s, attr)) for s, attr in series)
        for s, attr in series:
            h = getattr(s, attr)
            setattr(s, attr, [h[0] if h else 0] * (length - len(h)) + h)
