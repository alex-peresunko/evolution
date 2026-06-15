from .neural_network import NeuralNetwork


def save_brain(nn, filename, generation, genes, score, metrics=None, save_dir="saved_brains"):
    uptime = metrics.get_uptime_seconds() if metrics is not None else None
    nn.save(filename, save_dir=save_dir, generation=generation, genes=genes,
            score=score, uptime_seconds=uptime)


def load_brain(filename, target_topology, save_dir="saved_brains"):
    return NeuralNetwork.load(filename, target_topology, save_dir=save_dir)
