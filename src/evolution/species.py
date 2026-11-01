"""The two species and everything that differs between them as data."""
from dataclasses import dataclass, field

from . import config
from .world.diet import GrazingDiet, PredatorDiet


@dataclass(eq=False)
class Species:
    """A species is compared by identity; tests may override a field with monkeypatch."""

    name: str
    is_predator: bool
    default_genes: dict = field(repr=False)
    gene_bounds: dict = field(repr=False)
    start_count: int
    max_health: float
    health_per_food: float
    eat_ticks: int
    swim_stamina_cost: float
    diet: object = field(repr=False)

    @property
    def title(self):
        return self.name.capitalize()

    @property
    def plural(self):
        return f"{self.name}s"

    @property
    def brain_file(self):
        return f"best_brain_{self.name}.json"

    def __repr__(self):
        return f"Species({self.name})"


HERBIVORE = Species(
    name="herbivore",
    is_predator=False,
    default_genes=config.DEFAULT_HERBIVORE_GENES,
    gene_bounds=config.HERBIVORE_GENE_MIN_MAX,
    start_count=config.NUM_HERBIVOROUS,
    max_health=config.HERBIVORE_HEALTH,
    health_per_food=config.HERBIVORE_HEALTH_PER_FOOD,
    eat_ticks=config.HERBIVORE_EAT_TICKS,
    swim_stamina_cost=config.HERBIVORE_SWIM_STAMINA_COST,
    diet=GrazingDiet(),
)

CARNIVORE = Species(
    name="carnivore",
    is_predator=True,
    default_genes=config.DEFAULT_CARNIVORE_GENES,
    gene_bounds=config.CARNIVORE_GENE_MIN_MAX,
    start_count=config.NUM_CARNIVORES,
    max_health=config.CARNIVORE_HEALTH,
    health_per_food=config.CARNIVORE_HEALTH_PER_FOOD,
    eat_ticks=config.CARNIVORE_EAT_TICKS,
    swim_stamina_cost=config.CARNIVORE_SWIM_STAMINA_COST,
    diet=PredatorDiet(prey=HERBIVORE),
)

# Iteration order is the simulation's processing order (herbivores first); changing it changes
# every run.
ALL_SPECIES = (HERBIVORE, CARNIVORE)

_BY_NAME = {s.name: s for s in ALL_SPECIES}


def species_named(name):
    return _BY_NAME[name]
