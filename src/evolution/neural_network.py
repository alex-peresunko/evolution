import json
import os
import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            self.weights.append(w)
            b = np.random.randn(1, layer_sizes[i + 1]) * 0.1
            if i == len(layer_sizes) - 2:
                b[0, 0], b[0, 1] = 0.0, 1.0
            self.biases.append(b)

    def forward(self, inputs):
        x = np.array(inputs, dtype=float).reshape(1, -1)
        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                x = np.tanh(x)
        return np.tanh(x)[0]

    def mutate(self, mutation_rate, mutation_amount):
        for i in range(len(self.weights)):
            w_mask = np.random.rand(*self.weights[i].shape) < mutation_rate
            self.weights[i] += w_mask * np.random.uniform(
                -mutation_amount, mutation_amount, self.weights[i].shape
            )
            b_mask = np.random.rand(*self.biases[i].shape) < mutation_rate
            self.biases[i] += b_mask * np.random.uniform(
                -mutation_amount, mutation_amount, self.biases[i].shape
            )

    def copy(self):
        new_nn = NeuralNetwork(self.layer_sizes)
        new_nn.weights = [w.copy() for w in self.weights]
        new_nn.biases = [b.copy() for b in self.biases]
        return new_nn

    def save(self, filename, save_dir="saved_brains", generation=None, genes=None,
             score=None, uptime_seconds=None):
        os.makedirs(save_dir, exist_ok=True)
        data = {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        if generation is not None:
            data["generation"] = generation
        if score is not None:
            data["score"] = score
        if genes is not None:
            data["genes"] = {k: float(v) for k, v in genes.items()}
        if uptime_seconds is not None:
            data["metric_game_uptime_seconds"] = uptime_seconds
        with open(os.path.join(save_dir, filename), "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)
        print(f"Brain saved to {filename}")

    @staticmethod
    def load(filename, target_topology, save_dir="saved_brains"):
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Brain file not found at {filepath}")
        with open(filepath) as f:
            data = json.load(f)
        saved_weights = [np.array(w) for w in data["weights"]]
        saved_biases = [np.array(b) for b in data["biases"]]
        nn = NeuralNetwork(target_topology)
        for i in range(len(nn.weights)):
            if i < len(saved_weights):
                tw, sw = nn.weights[i].shape, saved_weights[i].shape
                tb, sb = nn.biases[i].shape, saved_biases[i].shape
                rows_w = min(tw[0], sw[0])
                cols_w = min(tw[1], sw[1])
                cols_b = min(tb[1], sb[1])
                nn.weights[i][:rows_w, :cols_w] = saved_weights[i][:rows_w, :cols_w]
                nn.biases[i][:, :cols_b] = saved_biases[i][:, :cols_b]
        print(f"Brain loaded from {filepath} into topology {target_topology}")
        return nn, data.get("generation", 1), data.get("genes")
