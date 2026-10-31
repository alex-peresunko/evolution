"""Prometheus metrics; the metric names are what grafana_dashboard.json queries."""
import time

from prometheus_client import Counter, Gauge

from ..config import BRAIN_TOPOLOGY
from ..species import ALL_SPECIES


class _SpeciesGauges:
    def __init__(self, species):
        name = species.name
        self.species = species
        self.best_score = Gauge(f"evolution_best_{name}_score", f"Best {name} score")
        self.count = Gauge(f"evolution_current_{name}_count", f"Current number of {name}s")
        self.best_gene = Gauge(f"evolution_best_{name}_gene", f"Best {name} gene value", ["gene"])
        self.best_stamina = Gauge(f"evolution_best_{name}_current_stamina",
                                  f"Best {name} current stamina")
        self.adult_count = Gauge(f"evolution_current_{name}_adult_count",
                                 f"Current number of adult {name}s")
        self.hall_of_fame_score = Gauge(f"evolution_hall_of_fame_{name}_score",
                                        f"All-time best {name} score")

    def update(self, population, hall_of_fame_score):
        best = max(population, key=lambda c: c.score, default=None)
        self.best_score.set(best.score if best else 0)
        self.count.set(len(population))
        self.best_stamina.set(best.stamina if best else 0)
        self.adult_count.set(sum(1 for c in population if c.is_adult))
        self.hall_of_fame_score.set(max(0, hall_of_fame_score))
        for gene, value in (best.genes if best else self.species.default_genes).items():
            self.best_gene.labels(gene).set(value)


class GameMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.uptime = Counter("evolution_running_seconds",
                              "Time the game has been running in seconds")
        self.generation = Gauge("evolution_current_generation_number",
                                "Current generation number")
        self.food_count = Gauge("evolution_current_food_count", "Current number of food items")
        self.species = [_SpeciesGauges(s) for s in ALL_SPECIES]
        Gauge("evolution_creature_brain_topology_input_size",
              "Brain input size").set(BRAIN_TOPOLOGY[0])
        Gauge("evolution_creature_brain_topology_hidden_layer_sizes",
              "Brain hidden layer count").set(len(BRAIN_TOPOLOGY) - 2)
        Gauge("evolution_creature_brain_topology_output_size",
              "Brain output size").set(BRAIN_TOPOLOGY[-1])
        Gauge("evolution_creature_brain_topology_hidden_layer_neurons",
              "Neurons in first hidden layer").set(BRAIN_TOPOLOGY[1])
        self.uptime.inc(0)

    def uptime_seconds(self):
        return time.time() - self.start_time

    def update(self, sim, dt):
        self.uptime.inc(dt)
        self.generation.set(sim.generation)
        self.food_count.set(len(sim.world.food))
        for gauges in self.species:
            gauges.update(sim.world.population(gauges.species),
                          sim.hall_of_fame[gauges.species].best.score)
