"""History of best brains: snapshots of best_brain_{herbivore,carnivore}.json kept as pairs in
saved_brains/history/<YYYYmmdd-HHMMSS>[-label]/, newest BRAIN_HISTORY_KEEP retained."""
import json
import os
import shutil
import time

from ..config import BRAIN_HISTORY_KEEP
from ..species import ALL_SPECIES
from .brain_store import SAVE_DIR


def history_dir(save_dir=SAVE_DIR):
    return os.path.join(save_dir, "history")


def snapshot_names(save_dir=SAVE_DIR):
    """Snapshot folder names, newest first (names start with a sortable timestamp)."""
    root = history_dir(save_dir)
    if not os.path.isdir(root):
        return []
    return sorted((n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n))),
                  reverse=True)


# Rewritten on every exit even when the brain is unchanged, so not part of a brain's identity.
_VOLATILE_KEYS = ("history", "metric_game_uptime_seconds")


def _brain_identity(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return ("unreadable", os.path.getsize(path))
    return {k: v for k, v in data.items() if k not in _VOLATILE_KEYS}


def _same_as(folder, save_dir):
    return all(_brain_identity(os.path.join(save_dir, sp.brain_file))
               == _brain_identity(os.path.join(folder, sp.brain_file)) for sp in ALL_SPECIES)


def new_snapshot_dir(label=None, save_dir=SAVE_DIR):
    """A fresh, empty snapshot folder (created)."""
    base = time.strftime("%Y%m%d-%H%M%S") + (f"-{label}" if label else "")
    name, n = base, 1
    while os.path.exists(os.path.join(history_dir(save_dir), name)):
        n += 1
        name = f"{base}-{n}"
    path = os.path.join(history_dir(save_dir), name)
    os.makedirs(path)
    return path


def snapshot_best(save_dir=SAVE_DIR, move=False, label=None):
    """Copy (or move) the current best brain files into a new snapshot. Returns its path, or
    None when there are no best files or they equal the newest snapshot (with move, the best
    files are still removed then, since that snapshot already holds them)."""
    present = [sp for sp in ALL_SPECIES if os.path.exists(os.path.join(save_dir, sp.brain_file))]
    if not present:
        return None
    names = snapshot_names(save_dir)
    path = None
    if not (names and _same_as(os.path.join(history_dir(save_dir), names[0]), save_dir)):
        path = new_snapshot_dir(label, save_dir)
        for sp in present:
            shutil.copy2(os.path.join(save_dir, sp.brain_file), path)
    if move:
        for sp in present:
            os.remove(os.path.join(save_dir, sp.brain_file))
    return path


def prune(keep=BRAIN_HISTORY_KEEP, save_dir=SAVE_DIR, protect=None):
    """Delete all but the newest `keep` snapshots; the folder `protect` is never deleted."""
    protect = os.path.abspath(protect) if protect else None
    for name in snapshot_names(save_dir)[keep:]:
        path = os.path.join(history_dir(save_dir), name)
        if os.path.abspath(path) != protect:
            shutil.rmtree(path)


def _summary(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("score"), data.get("generation")
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return None


def list_snapshots(save_dir=SAVE_DIR):
    """[(index, name, {species: (score, generation) or None})], index 1 = newest."""
    return [(i, name, {sp.name: _summary(os.path.join(history_dir(save_dir), name, sp.brain_file))
                       for sp in ALL_SPECIES})
            for i, name in enumerate(snapshot_names(save_dir), start=1)]


def resolve(spec, save_dir=SAVE_DIR):
    """Snapshot folder for an index (1 = newest) or a folder name. Raises ValueError."""
    names = snapshot_names(save_dir)
    if spec in names:
        name = spec
    elif str(spec).isdigit() and 1 <= int(spec) <= len(names):
        name = names[int(spec) - 1]
    else:
        raise ValueError(f"No brain snapshot {spec!r}; there are {len(names)} "
                         f"(see --list-brains).")
    return os.path.join(history_dir(save_dir), name)


def format_listing(save_dir=SAVE_DIR):
    rows = list_snapshots(save_dir)
    if not rows:
        return f"No brain snapshots in {history_dir(save_dir)}."
    lines = []
    for i, name, info in rows:
        parts = []
        for sp in info:
            if info[sp] is None:
                parts.append(f"{sp}: -")
            else:
                score, gen = info[sp]
                parts.append(f"{sp}: score {score if score is None else round(score, 3)}"
                             f" gen {gen}")
        lines.append(f"{i:3d}  {name:28s} " + "  ".join(parts))
    return "\n".join(lines)


def start_source(new=False, load=None, save_dir=SAVE_DIR):
    """Folder to build the starting populations from: None for random brains (new), the chosen
    snapshot (load), else save_dir itself (continue from best_brain_*.json)."""
    if new:
        return None
    if load is not None:
        return resolve(load, save_dir)
    return save_dir


def archive_after_start(source, new=False, save_dir=SAVE_DIR, keep=BRAIN_HISTORY_KEEP):
    """Snapshot the best files that were current before this run, then prune. Must run after the
    starting populations were read from source. A new or loaded run moves the best files out,
    so its own records are saved from scratch instead of having to beat the old ones."""
    path = snapshot_best(save_dir, move=new or source != save_dir)
    prune(keep, save_dir, protect=source if source not in (None, save_dir) else None)
    return path
