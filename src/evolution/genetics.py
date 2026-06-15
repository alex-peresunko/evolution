import random
import numpy as np

from .neural_network import NeuralNetwork
from .config import GENE_MIN_MAX, GENE_MUTATION_AMOUNT


def combine_brains(brain1: NeuralNetwork, brain2: NeuralNetwork) -> NeuralNetwork:
    child = brain1.copy()
    for i in range(len(child.weights)):
        mask = np.random.rand(*child.weights[i].shape) < 0.5
        child.weights[i][mask] = brain2.weights[i][mask]
        mask_b = np.random.rand(*child.biases[i].shape) < 0.5
        child.biases[i][mask_b] = brain2.biases[i][mask_b]
    return child


def combine_genes(genes1, genes2, default_genes, gene_constraints=None):
    constraints = gene_constraints or GENE_MIN_MAX
    child_genes = {}
    for key, default_value in default_genes.items():
        p1_gene = genes1.get(key, default_value)
        p2_gene = genes2.get(key, default_value)
        avg_gene = (p1_gene + p2_gene) / 2.0
        mutation = avg_gene * random.uniform(-GENE_MUTATION_AMOUNT, GENE_MUTATION_AMOUNT)
        min_val, max_val = constraints[key]
        child_genes[key] = float(np.clip(avg_gene + mutation, min_val, max_val))
    return child_genes


def mutate_genes(genes, default_genes, gene_constraints=None):
    constraints = gene_constraints or GENE_MIN_MAX
    mutated_genes = {}
    for key, default_value in default_genes.items():
        parent_value = genes.get(key, default_value)
        mutation = parent_value * random.uniform(-GENE_MUTATION_AMOUNT, GENE_MUTATION_AMOUNT)
        min_val, max_val = constraints[key]
        mutated_genes[key] = float(np.clip(parent_value + mutation, min_val, max_val))
    return mutated_genes
