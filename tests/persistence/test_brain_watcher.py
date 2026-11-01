import os

from evolution import config
from evolution.brain.network import NeuralNetwork
from evolution.persistence.brain_store import save_brain
from evolution.persistence.brain_watcher import BrainWatcher
from evolution.species import CARNIVORE, HERBIVORE


def _save(tmp_path, species, generation=1, score=1.0, history=None):
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    save_brain(brain, species.brain_file, save_dir=str(tmp_path), generation=generation,
               score=score, history=history)
    # Force a distinct mtime so the signature check sees the rewrite even on a coarse clock.
    stamp = os.stat(os.path.join(tmp_path, species.brain_file)).st_mtime + generation
    os.utime(os.path.join(tmp_path, species.brain_file), (stamp, stamp))


def test_files_present_at_startup_are_not_reported(tmp_path):
    _save(tmp_path, HERBIVORE)
    watcher = BrainWatcher(save_dir=str(tmp_path), interval=0)
    assert watcher.poll(0) == []


def test_new_brain_reports_only_that_species(tmp_path):
    _save(tmp_path, HERBIVORE)
    _save(tmp_path, CARNIVORE)
    watcher = BrainWatcher(save_dir=str(tmp_path), interval=0)
    _save(tmp_path, HERBIVORE, generation=2, score=5.0)
    [(species, data)] = watcher.poll(0)
    assert species is HERBIVORE
    assert data["generation"] == 2 and data["score"] == 5.0


def test_history_only_rewrite_is_ignored(tmp_path):
    brain = NeuralNetwork(config.BRAIN_TOPOLOGY)
    save_brain(brain, HERBIVORE.brain_file, save_dir=str(tmp_path), generation=1, score=1.0)
    watcher = BrainWatcher(save_dir=str(tmp_path), interval=0)
    save_brain(brain, HERBIVORE.brain_file, save_dir=str(tmp_path), generation=1, score=1.0,
               history={"score": [1.0, 2.0]})  # same brain, new chart history (as on exit)
    assert watcher.poll(0) == []


def test_missing_then_rewritten_file_is_reported(tmp_path):
    _save(tmp_path, HERBIVORE)
    watcher = BrainWatcher(save_dir=str(tmp_path), interval=0)
    os.remove(os.path.join(tmp_path, HERBIVORE.brain_file))  # headless --new moved it away
    assert watcher.poll(0) == []
    _save(tmp_path, HERBIVORE, generation=3)
    assert [species for species, _ in watcher.poll(0)] == [HERBIVORE]


def test_half_written_file_is_retried_next_poll(tmp_path):
    _save(tmp_path, HERBIVORE)
    watcher = BrainWatcher(save_dir=str(tmp_path), interval=0)
    path = os.path.join(tmp_path, HERBIVORE.brain_file)
    with open(path, "w") as f:
        f.write('{"weights": [')
    assert watcher.poll(0) == []
    _save(tmp_path, HERBIVORE, generation=4)
    assert [species for species, _ in watcher.poll(0)] == [HERBIVORE]


def test_polls_are_throttled_to_the_interval(tmp_path):
    _save(tmp_path, HERBIVORE)
    watcher = BrainWatcher(save_dir=str(tmp_path), interval=1.0)
    _save(tmp_path, HERBIVORE, generation=2)
    assert watcher.poll(0) != []  # first poll is always due
    _save(tmp_path, HERBIVORE, generation=3)
    assert watcher.poll(0.5) == []
    assert watcher.poll(1.0) != []
