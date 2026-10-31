import time

from prometheus_client import Counter, Gauge

from .config import BRAIN_TOPOLOGY, DEFAULT_CARNIVORE_GENES, DEFAULT_HERBIVORE_GENES


class GameMetrics:
    def __init__(self, sim_state):
        self.metric_game_uptime_seconds = Counter(
            "evolution_running_seconds", "Time the game has been running in seconds"
        )
        self.game_start_time = time.time()
        self.metric_current_generation_number = Gauge(
            "evolution_current_generation_number", "Current generation number"
        )
        self.metric_best_herbivore_score = Gauge(
            "evolution_best_herbivore_score", "Best herbivore score"
        )
        self.metric_best_carnivore_score = Gauge(
            "evolution_best_carnivore_score", "Best carnivore score"
        )
        self.metric_current_herbivore_count = Gauge(
            "evolution_current_herbivore_count", "Current number of herbivores"
        )
        self.metric_current_carnivore_count = Gauge(
            "evolution_current_carnivore_count", "Current number of carnivores"
        )
        self.metric_current_food_count = Gauge(
            "evolution_current_food_count", "Current number of food items"
        )
        self.metric_best_herbivore_gene = Gauge(
            "evolution_best_herbivore_gene", "Best herbivore gene value", ["gene"]
        )
        self.metric_best_carnivore_gene = Gauge(
            "evolution_best_carnivore_gene", "Best carnivore gene value", ["gene"]
        )
        self.metric_creature_brain_topology_input_size = Gauge(
            "evolution_creature_brain_topology_input_size", "Brain input size"
        )
        self.metric_creature_brain_topology_hidden_layers = Gauge(
            "evolution_creature_brain_topology_hidden_layer_sizes", "Brain hidden layer count"
        )
        self.metric_creature_brain_topology_output_size = Gauge(
            "evolution_creature_brain_topology_output_size", "Brain output size"
        )
        self.metric_creature_brain_topology_hidden_layer_neurons = Gauge(
            "evolution_creature_brain_topology_hidden_layer_neurons", "Neurons in first hidden layer"
        )
        self.metric_best_herbivore_current_stamina = Gauge(
            "evolution_best_herbivore_current_stamina", "Best herbivore current stamina"
        )
        self.metric_best_carnivore_current_stamina = Gauge(
            "evolution_best_carnivore_current_stamina", "Best carnivore current stamina"
        )
        self.metric_current_herbivore_adult_count = Gauge(
            "evolution_current_herbivore_adult_count", "Current number of adult herbivores"
        )
        self.metric_current_carnivore_adult_count = Gauge(
            "evolution_current_carnivore_adult_count", "Current number of adult carnivores"
        )
        self.metric_hall_of_fame_herbivore_score = Gauge(
            "evolution_hall_of_fame_herbivore_score", "All-time best herbivore score"
        )
        self.metric_hall_of_fame_carnivore_score = Gauge(
            "evolution_hall_of_fame_carnivore_score", "All-time best carnivore score"
        )

        self.metric_game_uptime_seconds.inc(0)
        self.metric_current_generation_number.set(0)
        self.metric_best_herbivore_score.set(0)
        self.metric_best_carnivore_score.set(0)
        self.metric_current_herbivore_count.set(len(sim_state["creatures"]))
        self.metric_current_carnivore_count.set(len(sim_state["carnivores"]))
        self.metric_current_food_count.set(len(sim_state["food_items"]))
        self.metric_best_herbivore_current_stamina.set(0)
        self.metric_best_carnivore_current_stamina.set(0)
        self.metric_current_herbivore_adult_count.set(0)
        self.metric_current_carnivore_adult_count.set(0)
        self.metric_hall_of_fame_herbivore_score.set(0)
        self.metric_hall_of_fame_carnivore_score.set(0)

        best_h = max(sim_state["creatures"], key=lambda c: c.score, default=None)
        best_c = max(sim_state["carnivores"], key=lambda c: c.score, default=None)
        for gene, value in (best_h.genes if best_h else DEFAULT_HERBIVORE_GENES).items():
            self.metric_best_herbivore_gene.labels(gene).set(value)
        for gene, value in (best_c.genes if best_c else DEFAULT_CARNIVORE_GENES).items():
            self.metric_best_carnivore_gene.labels(gene).set(value)
        self.metric_creature_brain_topology_input_size.set(BRAIN_TOPOLOGY[0])
        self.metric_creature_brain_topology_hidden_layers.set(len(BRAIN_TOPOLOGY) - 2)
        self.metric_creature_brain_topology_output_size.set(BRAIN_TOPOLOGY[-1])
        self.metric_creature_brain_topology_hidden_layer_neurons.set(BRAIN_TOPOLOGY[1])

    def get_uptime_seconds(self):
        return time.time() - self.game_start_time

    def update(self, sim_state, dt):
        best_h = max(sim_state["creatures"], key=lambda c: c.score, default=None)
        best_c = max(sim_state["carnivores"], key=lambda c: c.score, default=None)
        self.metric_game_uptime_seconds.inc(dt)
        self.metric_current_generation_number.set(sim_state["generation"])
        self.metric_best_herbivore_score.set(best_h.score if best_h else 0)
        self.metric_best_carnivore_score.set(best_c.score if best_c else 0)
        self.metric_current_herbivore_count.set(len(sim_state["creatures"]))
        self.metric_current_carnivore_count.set(len(sim_state["carnivores"]))
        self.metric_current_food_count.set(len(sim_state["food_items"]))
        self.metric_best_herbivore_current_stamina.set(best_h.stamina if best_h else 0)
        self.metric_best_carnivore_current_stamina.set(best_c.stamina if best_c else 0)
        self.metric_current_herbivore_adult_count.set(
            sum(1 for c in sim_state["creatures"] if c.is_adult)
        )
        self.metric_current_carnivore_adult_count.set(
            sum(1 for c in sim_state["carnivores"] if c.is_adult)
        )
        self.metric_hall_of_fame_herbivore_score.set(
            max(0, sim_state["hall_of_fame"]["herbivore"]["score"])
        )
        self.metric_hall_of_fame_carnivore_score.set(
            max(0, sim_state["hall_of_fame"]["carnivore"]["score"])
        )
        for gene, value in (best_h.genes if best_h else DEFAULT_HERBIVORE_GENES).items():
            self.metric_best_herbivore_gene.labels(gene).set(value)
        for gene, value in (best_c.genes if best_c else DEFAULT_CARNIVORE_GENES).items():
            self.metric_best_carnivore_gene.labels(gene).set(value)
        self.metric_creature_brain_topology_input_size.set(BRAIN_TOPOLOGY[0])
        self.metric_creature_brain_topology_hidden_layers.set(len(BRAIN_TOPOLOGY) - 2)
        self.metric_creature_brain_topology_output_size.set(BRAIN_TOPOLOGY[-1])
        self.metric_creature_brain_topology_hidden_layer_neurons.set(BRAIN_TOPOLOGY[1])
