import random
import numpy as np

from .neural_network import NeuralNetwork
from .config import (ADDITIVE_GENE_MUTATION, GENE_MIN_MAX, GENE_MUTATION_AMOUNT, MUTATION_AMOUNT,
                     MUTATION_RATE, SPECIATION_DISTANCE)


def combine_brains(brain1: NeuralNetwork, brain2: NeuralNetwork) -> NeuralNetwork:
    child = brain1.copy()
    for i in range(len(child.weights)):
        mask = np.random.rand(*child.weights[i].shape) < 0.5
        child.weights[i][mask] = brain2.weights[i][mask]
        mask_b = np.random.rand(*child.biases[i].shape) < 0.5
        child.biases[i][mask_b] = brain2.biases[i][mask_b]
    return child


def mutate_brain(brain: NeuralNetwork, genes) -> None:
    """Mutate in place at the rate and amount carried in the creature's own genes."""
    brain.mutate(genes.get("mutation_rate", MUTATION_RATE),
                 genes.get("mutation_amount", MUTATION_AMOUNT))


def _mutation(key, value):
    if key in ADDITIVE_GENE_MUTATION:
        amount = ADDITIVE_GENE_MUTATION[key]
        return random.uniform(-amount, amount)
    return value * random.uniform(-GENE_MUTATION_AMOUNT, GENE_MUTATION_AMOUNT)


def combine_genes(genes1, genes2, default_genes, gene_constraints=None):
    constraints = gene_constraints or GENE_MIN_MAX
    child_genes = {}
    for key, default_value in default_genes.items():
        p1_gene = genes1.get(key, default_value)
        p2_gene = genes2.get(key, default_value)
        avg_gene = (p1_gene + p2_gene) / 2.0
        min_val, max_val = constraints[key]
        child_genes[key] = float(np.clip(avg_gene + _mutation(key, avg_gene), min_val, max_val))
    return child_genes


def mutate_genes(genes, default_genes, gene_constraints=None):
    constraints = gene_constraints or GENE_MIN_MAX
    mutated_genes = {}
    for key, default_value in default_genes.items():
        parent_value = genes.get(key, default_value)
        min_val, max_val = constraints[key]
        mutated_genes[key] = float(np.clip(parent_value + _mutation(key, parent_value),
                                           min_val, max_val))
    return mutated_genes


def genetic_distance(genes1, genes2, gene_constraints=None):
    """RMS difference of the genes both carry, each scaled by its allowed range."""
    constraints = gene_constraints or GENE_MIN_MAX
    diffs = [((genes1[k] - genes2[k]) / (hi - lo)) ** 2
             for k, (lo, hi) in constraints.items() if k in genes1 and k in genes2]
    return (sum(diffs) / len(diffs)) ** 0.5 if diffs else 0.0


def compatible(genes1, genes2, gene_constraints=None):
    """Whether two creatures are close enough genetically to breed (reproductive isolation)."""
    return genetic_distance(genes1, genes2, gene_constraints) <= SPECIATION_DISTANCE
