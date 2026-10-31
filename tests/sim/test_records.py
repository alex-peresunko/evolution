import pytest

from evolution import config
from evolution.brain.network import NeuralNetwork
from evolution.sim.records import HallOfFame, Record, SavedScore, SpeciesRecords
from evolution.species import CARNIVORE, HERBIVORE
from evolution.world.creature import Creature

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _dead(score):
    c = Creature(HERBIVORE, WORLD_BOUNDS)
    c.score = score
    return c


def test_new_best_is_a_fresh_copy_with_its_origin():
    records = SpeciesRecords()
    c = _dead(3.0)
    assert records.retire(c)
    best = records.best
    assert best.score == 3.0 and best.fresh
    assert best.brain is not c.brain and best.genes is not c.genes
    assert best.source == f"creature #{c.id}, age {c.age}, lineage #{c.lineage}"


def test_worse_creature_enters_the_pool_but_not_the_best():
    records = SpeciesRecords()
    records.retire(_dead(3.0))
    assert not records.retire(_dead(1.0))
    assert records.best.score == 3.0
    assert sorted(r.score for r in records.elites) == [1.0, 3.0]


def test_seed_is_saved_and_not_fresh():
    records = SpeciesRecords()
    records.seed(Record(7.0, NeuralNetwork(config.BRAIN_TOPOLOGY), {}, 4))
    assert records.has_brain and not records.best.fresh
    assert records.saved == SavedScore(7.0, 7.0)
    assert records.elites[0] is not records.best


def test_fade_scales_scores_but_not_the_saved_raw_score():
    records = SpeciesRecords()
    records.seed(Record(8.0, NeuralNetwork(config.BRAIN_TOPOLOGY), {}, 1))
    records.fade(0.5)
    assert records.best.score == records.elites[0].score == pytest.approx(4.0)
    assert records.saved == SavedScore(8.0, pytest.approx(4.0))


def test_empty_record_does_not_fade_below_zero():
    records = SpeciesRecords()
    records.fade(0.5)
    assert records.best.score == -1


def test_hall_of_fame_is_per_species():
    hof = HallOfFame()
    hof[HERBIVORE].retire(_dead(2.0))
    assert hof[HERBIVORE].has_brain and not hof[CARNIVORE].has_brain
