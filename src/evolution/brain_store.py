import json
import os

from .neural_network import NeuralNetwork


def save_brain(nn, filename, generation, genes, score, metrics=None, save_dir="saved_brains",
               history=None):
    uptime = metrics.get_uptime_seconds() if metrics is not None else None
    nn.save(filename, save_dir=save_dir, generation=generation, genes=genes,
            score=score, uptime_seconds=uptime, history=history)


def read_saved_score(filename, save_dir="saved_brains"):
    """Score stored in a saved brain file, or None if missing/unreadable/unscored."""
    try:
        with open(os.path.join(save_dir, filename)) as f:
            return json.load(f).get("score")
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return None


def read_saved_history(filename, save_dir="saved_brains"):
    """Chart history ({"score": [...], "longevity": [...]}) stored in a saved brain file, or None."""
    try:
        with open(os.path.join(save_dir, filename)) as f:
            return json.load(f).get("history")
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return None


def update_saved_history(filename, history, save_dir="saved_brains"):
    """Replace only the chart history of an existing brain file. Returns True if written."""
    path = os.path.join(save_dir, filename)
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    data["history"] = history
    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)
    return True


def save_brain_if_better(nn, filename, generation, genes, score, metrics=None,
                         save_dir="saved_brains", history=None):
    """save_brain only if score beats the file's saved score. Returns True if written."""
    prev = read_saved_score(filename, save_dir=save_dir)
    if prev is not None and score <= prev:
        return False
    save_brain(nn, filename, generation, genes, score, metrics=metrics, save_dir=save_dir,
               history=history)
    return True


def load_brain(filename, target_topology, save_dir="saved_brains"):
    return NeuralNetwork.load(filename, target_topology, save_dir=save_dir)
