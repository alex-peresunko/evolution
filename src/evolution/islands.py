"""Island model for background mode: game.py runs worker copies of game_headless.py, and the
main sim and the workers swap their best brains through files in saved_brains/.

Each worker writes its Hall of Fame to worker_{id}_{species}.json; the main sim imports any that
beats its own and publishes its Hall of Fame (the best across all islands) to
worker_main_{species}.json, which every worker imports in turn."""
import glob
import json
import os

from .brain_store import save_brain
from .config import (
    BRAIN_TOPOLOGY,
    CARNIVORE_GENE_MIN_MAX,
    DEFAULT_CARNIVORE_GENES,
    DEFAULT_HERBIVORE_GENES,
    GENE_MIN_MAX,
    ISLAND_IMMIGRANTS,
    RESPAWN_IMMUNITY_TICKS,
)
from .entities import Creature
from .genetics import mutate_brain, mutate_genes
from .neural_network import NeuralNetwork

SAVE_DIR = "saved_brains"
SPECIES = ("herbivore", "carnivore")


def worker_file(worker_id, species):
    return f"worker_{worker_id}_{species}.json"


def main_file(species):
    return f"worker_main_{species}.json"


def clear_island_files(save_dir=SAVE_DIR):
    """Remove exchange files left by a previous run (they may come from an older config)."""
    for path in glob.glob(os.path.join(save_dir, "worker_*.json*")):  # incl. stray .tmp files
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def inject_immigrants(sim_state, species, brain, genes, count=ISLAND_IMMIGRANTS, generation=1):
    """Replace the lowest-scoring creatures of a species with the brain and mutated copies."""
    is_carnivore = species == "carnivore"
    key = "carnivores" if is_carnivore else "creatures"
    default_genes = DEFAULT_CARNIVORE_GENES if is_carnivore else DEFAULT_HERBIVORE_GENES
    constraints = CARNIVORE_GENE_MIN_MAX if is_carnivore else GENE_MIN_MAX
    template = (sim_state[key] or sim_state["creatures"] + sim_state["carnivores"] or [None])[0]
    kwargs = ({"vec2_factory": template._vec2, "rect_factory": template._rect}
              if template is not None else {})
    replaced = {id(c) for c in sorted(sim_state[key], key=lambda c: c.score)[:count]}
    immigrants = []
    for i in range(count):
        b, g = brain.copy(), dict(genes)
        if i:
            g = mutate_genes(genes, default_genes, gene_constraints=constraints)
            mutate_brain(b, g)
        c = Creature(sim_state["world_bounds"], brain=b, is_carnivore=is_carnivore, genes=g,
                     **kwargs)
        c.respawn_immunity_ticks = RESPAWN_IMMUNITY_TICKS
        c.generation = generation
        immigrants.append(c)
    sim_state[key] = [c for c in sim_state[key] if id(c) not in replaced] + immigrants


def import_if_better(sim_state, species, filename, seen_mtimes, save_dir=SAVE_DIR):
    """If the file changed since last seen and its score beats this sim's Hall of Fame, adopt it:
    the Hall of Fame takes the brain and immigrants carrying it replace the weakest creatures.
    Returns (old_score, new_score) on import, else None."""
    path = os.path.join(save_dir, filename)
    try:
        mtime = os.path.getmtime(path)
        if mtime <= seen_mtimes.get(filename, 0):
            return None
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None  # missing, or mid-replace on some platforms: retried next sync
    seen_mtimes[filename] = mtime
    score = data.get("score")
    hof = sim_state["hall_of_fame"][species]
    if score is None or score <= hof["score"] or "weights" not in data:
        return None
    brain = NeuralNetwork.from_data(data, BRAIN_TOPOLOGY)
    default = DEFAULT_CARNIVORE_GENES if species == "carnivore" else DEFAULT_HERBIVORE_GENES
    genes = {**default, **(data.get("genes") or {})}
    old_score = hof["score"]
    generation = data.get("generation", 1)
    sim_state["hall_of_fame"][species] = {"score": score, "brain": brain, "genes": genes,
                                          "generation": generation, "fresh": True,
                                          "source": f"imported from {filename}"}
    inject_immigrants(sim_state, species, brain, genes, generation=generation)
    return old_score, score


def publish_hall_of_fame(sim_state, published, save_dir=SAVE_DIR):
    """Write each species' Hall of Fame to its main_file when it changed since last published."""
    for species in SPECIES:
        hof = sim_state["hall_of_fame"][species]
        if hof["brain"] is None or hof["brain"] is published.get(species):
            continue
        save_brain(hof["brain"], main_file(species), generation=sim_state["generation"],
                   genes=hof["genes"], score=hof["score"], save_dir=save_dir)
        published[species] = hof["brain"]
