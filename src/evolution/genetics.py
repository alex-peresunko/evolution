"""Inheritance: brain crossover and mutation, gene mixing, and the speciation test."""
import random

import numpy as np

from .config import (
    ADDITIVE_GENE_MUTATION,
    GENE_MUTATION_AMOUNT,
    MUTATION_AMOUNT,
    MUTATION_RATE,
    SPECIATION_DISTANCE,
)


def combine_brains(brain1, brain2):
    """Each neuron's incoming weights and bias come from one parent, as a unit."""
    child = brain1.copy()
    for i in range(len(child.weights)):
        from_2 = np.random.rand(child.weights[i].shape[1]) < 0.5
        child.weights[i][:, from_2] = brain2.weights[i][:, from_2]
        child.biases[i][:, from_2] = brain2.biases[i][:, from_2]
    return child


def mutate_brain(brain, genes):
    """Mutate in place at the rate and amount carried in the creature's own genes."""
    brain.mutate(genes.get("mutation_rate", MUTATION_RATE),
                 genes.get("mutation_amount", MUTATION_AMOUNT))


def _clamp(value, bounds):
    return float(np.clip(value, *bounds))


def clamp_genes(genes, species):
    """genes with every value the species bounds limit clamped into them (in place)."""
    for key, value in genes.items():
        if key in species.gene_bounds:
            genes[key] = _clamp(value, species.gene_bounds[key])
    return genes


def _mutation(key, value):
    if key in ADDITIVE_GENE_MUTATION:
        amount = ADDITIVE_GENE_MUTATION[key]
        return random.uniform(-amount, amount)
    return value * random.uniform(-GENE_MUTATION_AMOUNT, GENE_MUTATION_AMOUNT)


def combine_genes(genes1, genes2, species):
    """The parents' mean of each of the species' genes, mutated; missing genes take the
    species default."""
    child_genes = {}
    for key, default_value in species.default_genes.items():
        avg_gene = (genes1.get(key, default_value) + genes2.get(key, default_value)) / 2.0
        child_genes[key] = _clamp(avg_gene + _mutation(key, avg_gene), species.gene_bounds[key])
    return child_genes


def mutate_genes(genes, species):
    mutated_genes = {}
    for key, default_value in species.default_genes.items():
        parent_value = genes.get(key, default_value)
        mutated_genes[key] = _clamp(parent_value + _mutation(key, parent_value),
                                    species.gene_bounds[key])
    return mutated_genes


def genetic_distance(genes1, genes2, gene_bounds):
    """RMS difference of the genes both carry, each scaled by its allowed range."""
    diffs = [((genes1[k] - genes2[k]) / (hi - lo)) ** 2
             for k, (lo, hi) in gene_bounds.items() if k in genes1 and k in genes2]
    return (sum(diffs) / len(diffs)) ** 0.5 if diffs else 0.0


def compatible(genes1, genes2, species):
    """Whether two creatures are close enough genetically to breed (reproductive isolation)."""
    return genetic_distance(genes1, genes2, species.gene_bounds) <= SPECIATION_DISTANCE
