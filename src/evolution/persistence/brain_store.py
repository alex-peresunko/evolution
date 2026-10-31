"""Brain files: JSON with the network weights plus generation, genes, score and chart history.
The format is shared with every saved_brains/ file in git, so keys must stay compatible."""
import json
import os

from ..brain.network import NeuralNetwork
from ..config import BRAIN_TOPOLOGY

SAVE_DIR = "saved_brains"


def write_json_atomic(path, data):
    """Write to a temp file and rename it over path, so concurrent readers (the island sync in
    game.py) never see a half-written file."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)
    os.replace(tmp, path)


def save_brain(brain, filename, *, generation=None, genes=None, score=None, uptime_seconds=None,
               history=None, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    data = brain.to_data()
    if generation is not None:
        data["generation"] = generation
    if score is not None:
        data["score"] = score
    if genes is not None:
        data["genes"] = {k: float(v) for k, v in genes.items()}
    if uptime_seconds is not None:
        data["metric_game_uptime_seconds"] = uptime_seconds
    if history is not None:
        data["history"] = history
    write_json_atomic(os.path.join(save_dir, filename), data)
    print(f"Brain saved to {filename}")


def load_brain(filename, topology=BRAIN_TOPOLOGY, save_dir=SAVE_DIR):
    """(brain, generation, genes or None); raises FileNotFoundError."""
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Brain file not found at {path}")
    with open(path) as f:
        data = json.load(f)
    brain = NeuralNetwork.from_data(data, topology)
    print(f"Brain loaded from {path} into topology {topology}")
    return brain, data.get("generation", 1), data.get("genes")


def _read(filename, key, save_dir):
    try:
        with open(os.path.join(save_dir, filename)) as f:
            return json.load(f).get(key)
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return None


def read_saved_score(filename, save_dir=SAVE_DIR):
    """Score stored in a saved brain file, or None if missing/unreadable/unscored."""
    return _read(filename, "score", save_dir)


def read_saved_history(filename, save_dir=SAVE_DIR):
    """Chart history ({"score": [...], "longevity": [...]}) stored in a saved brain file, or None."""
    return _read(filename, "history", save_dir)


def update_saved_history(filename, history, save_dir=SAVE_DIR):
    """Replace only the chart history of an existing brain file. Returns True if written."""
    path = os.path.join(save_dir, filename)
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    data["history"] = history
    write_json_atomic(path, data)
    return True
