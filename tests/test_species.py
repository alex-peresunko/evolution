import pytest

from evolution import config
from evolution.species import ALL_SPECIES, CARNIVORE, HERBIVORE, species_named


def test_herbivores_are_processed_first():
    assert ALL_SPECIES == (HERBIVORE, CARNIVORE)


@pytest.mark.parametrize("species", ALL_SPECIES)
def test_lookup_by_name_and_file_names(species):
    assert species_named(species.name) is species
    assert species.brain_file == f"best_brain_{species.name}.json"


def test_fields_mirror_the_config_constants():
    assert HERBIVORE.default_genes is config.DEFAULT_HERBIVORE_GENES
    assert HERBIVORE.gene_bounds is config.HERBIVORE_GENE_MIN_MAX
    assert CARNIVORE.gene_bounds is config.CARNIVORE_GENE_MIN_MAX
    assert (HERBIVORE.start_count, CARNIVORE.start_count) == (
        config.NUM_HERBIVOROUS, config.NUM_CARNIVORES)
    assert CARNIVORE.is_predator and not HERBIVORE.is_predator
    assert CARNIVORE.diet.prey is HERBIVORE
