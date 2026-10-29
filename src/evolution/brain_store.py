import json
import os

from .neural_network import NeuralNetwork


def save_brain(nn, filename, generation, genes, score, metrics=None, save_dir="saved_brains"):
    uptime = metrics.get_uptime_seconds() if metrics is not None else None
    nn.save(filename, save_dir=save_dir, generation=generation, genes=genes,
            score=score, uptime_seconds=uptime)


def read_saved_score(filename, save_dir="saved_brains"):
    """Score stored in a saved brain file, or None if missing/unreadable/unscored."""
    try:
        with open(os.path.join(save_dir, filename)) as f:
            return json.load(f).get("score")
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return None


def save_brain_if_better(nn, filename, generation, genes, score, metrics=None,
                         save_dir="saved_brains"):
    """save_brain only if score beats the file's saved score. Returns True if written."""
    prev = read_saved_score(filename, save_dir=save_dir)
    if prev is not None and score <= prev:
        return False
    save_brain(nn, filename, generation, genes, score, metrics=metrics, save_dir=save_dir)
    return True


def load_brain(filename, target_topology, save_dir="saved_brains"):
    return NeuralNetwork.load(filename, target_topology, save_dir=save_dir)
