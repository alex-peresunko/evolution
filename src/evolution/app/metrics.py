"""Prometheus metrics, computed from the simulation when Prometheus scrapes. The names and labels
are what monitoring/dashboard.py queries."""
import statistics
import time

import numpy as np
from prometheus_client import REGISTRY, start_http_server
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

from ..config import BRAIN_TOPOLOGY, FAT_MAX, SPECIATION_DISTANCE
from ..species import ALL_SPECIES


def genetic_diversity(population, species):
    """RMS genetic_distance between two random members (same units as SPECIATION_DISTANCE)."""
    if len(population) < 2:
        return 0.0
    bounds = species.gene_bounds
    genes = [g for g in bounds if g in population[0].genes]
    values = np.array([[c.genes[g] for g in genes] for c in population])
    ranges = np.array([bounds[g][1] - bounds[g][0] for g in genes])
    return float(np.sqrt(np.mean(2 * values.var(axis=0) / ranges ** 2)))


def _family(kind, name, doc, labels=()):
    cls = CounterMetricFamily if kind == "counter" else GaugeMetricFamily
    return cls(f"evolution_{name}", doc, labels=list(labels))


class GameMetrics:
    """Serves the metrics of the simulation it watches; the work happens on the scrape thread,
    so the simulation loop pays nothing for them."""

    def __init__(self):
        self.start_time = time.time()
        self.sim = None
        self.autosaver = None

    def uptime_seconds(self):
        return time.time() - self.start_time

    def watch(self, sim, autosaver=None):
        self.sim, self.autosaver = sim, autosaver

    def serve(self, port):
        REGISTRY.register(self)
        start_http_server(port)

    def collect(self):
        uptime = _family("counter", "running_seconds", "Time the simulation has been running")
        uptime.add_metric([], self.uptime_seconds())
        yield uptime
        topology = _family("gauge", "brain_layer_neurons", "Neurons per brain layer", ["layer"])
        for i, size in enumerate(BRAIN_TOPOLOGY):
            topology.add_metric([str(i)], size)
        yield topology
        speciation = _family("gauge", "speciation_distance",
                             "Largest genetic distance at which two creatures can still breed")
        speciation.add_metric([], SPECIATION_DISTANCE)
        yield speciation
        if self.sim is not None:
            yield from self._world(self.sim)
            yield from self._species(self.sim)
        if self.autosaver is not None:
            yield from self._saves(self.sim, self.autosaver)

    @staticmethod
    def _world(sim):
        world, tally = sim.world, sim.world.tally
        for kind, name, doc, value in (
                ("counter", "ticks", "Simulation ticks run", sim.tick),
                ("gauge", "generation", "Deepest generation born", sim.generation),
                ("gauge", "food", "Food items in the world", len(world.food)),
                ("gauge", "season", "Season: -1 midwinter .. 1 midsummer", world.season),
                ("gauge", "daylight", "Daylight: 0 night .. 1 day", world.light),
                ("counter", "bites", "Carnivore bites that landed", tally.bites)):
            family = _family(kind, name, doc)
            family.add_metric([], value)
            yield family
        food = _family("counter", "food_removed", "Food items removed, by fate", ["fate"])
        food.add_metric(["eaten"], tally.food_eaten)
        food.add_metric(["rotted"], tally.food_rotted)
        carcasses = _family("counter", "carcasses_removed", "Carcasses removed, by fate", ["fate"])
        carcasses.add_metric(["eaten"], tally.carcasses_eaten)
        carcasses.add_metric(["rotted"], tally.carcasses_rotted)
        yield food
        yield carcasses

    @staticmethod
    def _species(sim):
        sp = ["species"]
        gauges = {name: _family("gauge", name, doc, sp) for name, doc in (
            ("population", "Living creatures"),
            ("adults", "Living adults"),
            ("hall_of_fame_score", "Hall of Fame best score (fades over time)"),
            ("max_age_ticks", "Oldest age reached at death this run"),
            ("lineages", "Founder lineages still alive"),
            ("genetic_diversity", "RMS genetic distance between two random members"),
            ("health_ratio", "Mean health / max health"),
            ("stamina_ratio", "Mean stamina / max stamina"),
            ("fat_ratio", "Mean fat / FAT_MAX"),
            ("in_water_share", "Share of creatures swimming"),
            ("eating_share", "Share of creatures standing still to eat"),
            ("winded_share", "Share of creatures out of stamina"),
            ("signal_mean", "Mean signal output"))}
        counters = {name: _family("counter", name, doc, sp) for name, doc in (
            ("births", "Creatures born"),
            ("extinctions", "Times the species died out and was repopulated"),
            ("death_age_ticks", "Sum of ages at death; divide by deaths for mean lifespan"))}
        score = _family("gauge", "score", "Score of living creatures", ["species", "stat"])
        deaths = _family("counter", "deaths", "Deaths, by likely cause", ["species", "cause"])
        gene_mean = _family("gauge", "gene_mean", "Population mean gene value",
                            ["species", "gene"])
        gene_drift = _family("gauge", "gene_drift", "Population mean gene / species default",
                             ["species", "gene"])
        best_gene = _family("gauge", "best_gene", "Gene value of the best living creature",
                            ["species", "gene"])

        for species in ALL_SPECIES:
            population = list(sim.world.population(species))
            stats = sim.stats[species]
            name = species.name
            n = len(population)

            def mean(values, n=n):
                return sum(values) / n if n else 0.0

            for key, value in (
                    ("population", n),
                    ("adults", sum(1 for c in population if c.is_adult)),
                    ("hall_of_fame_score", max(0.0, sim.hall_of_fame[species].best.score)),
                    ("max_age_ticks", stats.max_age),
                    ("lineages", len({c.lineage for c in population})),
                    ("genetic_diversity", genetic_diversity(population, species)),
                    ("health_ratio", mean(c.health / c.max_health for c in population)),
                    ("stamina_ratio", mean(c.stamina / c.max_stamina for c in population)),
                    ("fat_ratio", mean(c.fat / FAT_MAX for c in population)),
                    ("in_water_share", mean(c.in_water for c in population)),
                    ("eating_share", mean(c.eating_ticks > 0 for c in population)),
                    ("winded_share", mean(c.winded for c in population)),
                    ("signal_mean", mean(c.signal for c in population))):
                gauges[key].add_metric([name], value)
            counters["births"].add_metric([name], stats.births)
            counters["extinctions"].add_metric([name], stats.extinctions)
            counters["death_age_ticks"].add_metric([name], stats.death_age_total)
            for cause, count in stats.deaths.items():
                deaths.add_metric([name, cause], count)

            scores = [c.score for c in population]
            for stat, value in (("best", max(scores, default=0.0)),
                                ("mean", mean(scores)),
                                ("median", statistics.median(scores) if scores else 0.0)):
                score.add_metric([name, stat], value)

            best = max(population, key=lambda c: c.score, default=None)
            for gene, default in species.default_genes.items():
                avg = mean(c.genes[gene] for c in population) if n else default
                gene_mean.add_metric([name, gene], avg)
                gene_drift.add_metric([name, gene], avg / default if default else 0.0)
                best_gene.add_metric([name, gene], best.genes[gene] if best else default)

        yield from gauges.values()
        yield from counters.values()
        yield from (score, deaths, gene_mean, gene_drift, best_gene)

    @staticmethod
    def _saves(sim, autosaver):
        saves = _family("counter", "brain_saves", "Best brains written to disk", ["species"])
        saved = _family("gauge", "saved_score", "Score of the brain last written to disk",
                        ["species"])
        last = _family("gauge", "last_save_timestamp_seconds",
                       "When the species' brain was last written", ["species"])
        for species in ALL_SPECIES:
            saves.add_metric([species.name], autosaver.saves[species])
            if sim is not None and sim.hall_of_fame[species].saved is not None:
                saved.add_metric([species.name], sim.hall_of_fame[species].saved.raw)
            if species in autosaver.last_save_time:
                last.add_metric([species.name], autosaver.last_save_time[species])
        yield from (saves, saved, last)
