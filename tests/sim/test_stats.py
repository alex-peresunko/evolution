from evolution.sim.stats import SpeciesStats, Statistics
from evolution.species import CARNIVORE, HERBIVORE


def test_restore_left_pads_every_series_to_the_longest():
    stats = Statistics()
    stats.restore(HERBIVORE, {"score": [1.0, 2.0, 3.0], "longevity": [10, 20]})
    herb, carn = stats[HERBIVORE], stats[CARNIVORE]
    assert herb.score_history == [1.0, 2.0, 3.0]
    assert herb.longevity_history == [10, 10, 20]
    assert carn.score_history == carn.longevity_history == [0, 0, 0]
    assert herb.max_age == 20


def test_saved_history_round_trips():
    stats = SpeciesStats()
    stats.sample(5.0)
    stats.restore(stats.saved_history())
    assert stats.saved_history() == {"score": [5.0], "longevity": [0]}
