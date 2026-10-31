from evolution import config
from evolution.persistence.autosave import HallOfFameAutosaver, save_on_exit
from evolution.persistence.brain_store import read_saved_score
from evolution.sim.simulation import Simulation
from evolution.species import CARNIVORE, HERBIVORE
from evolution.world.creature import Creature
from evolution.world.world import World

WORLD_BOUNDS = (config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _sim(save_dir, **kwargs):
    autosaver = HallOfFameAutosaver(save_dir=save_dir, **kwargs)
    return Simulation(World(WORLD_BOUNDS, {}, []), listeners=[autosaver]), autosaver


def _retire(sim, score, age=0):
    c = Creature(HERBIVORE, WORLD_BOUNDS)
    c.score, c.age = score, age
    sim.retire(c)
    return c


def test_hall_of_fame_save_logs_score_change_and_creature(tmp_path, capsys):
    sim, _ = _sim(tmp_path)
    for score in (2.0, 5.5):
        sim.world.tick += config.AUTOSAVE_INTERVAL
        c = _retire(sim, score, age=1234)
    out = capsys.readouterr().out
    assert "[HoF save] Herbivore best brain saved to best_brain_herbivore.json: 2.000 (new)" in out
    assert f"2.000 -> 5.500 (+3.500); gen 1, creature #{c.id}, age 1234, lineage #{c.lineage}" in out
    assert read_saved_score(HERBIVORE.brain_file, save_dir=tmp_path) == 5.5


def test_hall_of_fame_save_log_shows_the_faded_previous_score(tmp_path, capsys):
    sim, _ = _sim(tmp_path)
    for score in (8.0, 5.0):
        sim.world.tick += config.AUTOSAVE_INTERVAL
        _retire(sim, score)
        sim.hall_of_fame.fade(0.5)
    assert "8.000 (faded to 4.000) -> 5.000 (+1.000)" in capsys.readouterr().out


def test_record_inside_the_interval_waits_for_a_later_save(tmp_path):
    sim, autosaver = _sim(tmp_path)
    sim.world.tick = config.AUTOSAVE_INTERVAL
    _retire(sim, 1.0)
    _retire(sim, 2.0)
    assert autosaver.unsaved == {HERBIVORE}
    assert read_saved_score(HERBIVORE.brain_file, save_dir=tmp_path) == 1.0
    sim.world.tick += config.AUTOSAVE_INTERVAL
    autosaver.deaths_processed(sim)
    assert not autosaver.unsaved
    assert read_saved_score(HERBIVORE.brain_file, save_dir=tmp_path) == 2.0


def test_worker_saves_to_its_exchange_file(tmp_path):
    sim, autosaver = _sim(tmp_path, worker_id=3)
    sim.world.tick = config.AUTOSAVE_INTERVAL
    _retire(sim, 1.0)
    assert (tmp_path / autosaver.filename(HERBIVORE)).exists()
    assert not (tmp_path / HERBIVORE.brain_file).exists()


def test_exit_save_writes_only_fresh_records(tmp_path, capsys):
    sim, _ = _sim(tmp_path)
    sim.listeners.clear()
    _retire(sim, 3.0)
    save_on_exit(sim, save_dir=tmp_path)
    assert read_saved_score(HERBIVORE.brain_file, save_dir=tmp_path) == 3.0
    assert not (tmp_path / CARNIVORE.brain_file).exists()
    assert "No carnivore in Hall of Fame to save." in capsys.readouterr().out
