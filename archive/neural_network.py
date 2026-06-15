import random
import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 4],
     [3, 4, 5, 6],
     [-3, -2.4, 5, 2]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



layer1 = Layer_Dense(4, 10)
layer2 = Layer_Dense(10, 2)

layer1.forward(X)

print(layer1.output)

layer2.forward(layer1.output)

print(layer2.output)
exit()

class Neuron:
    def __init__(self, synapses_qty=1, weight_list: list=None, bias=0):
        self.bias = bias
        self.axon_signal = 0
        self.synapses_qty = synapses_qty
        self.synapse_signals = [0, 0, 0]
        self.synapses_weights = []
        self._calculated = False
        self.weight_total = 0
        for w in range(1, self.synapses_qty):
            if weight_list is None:
                self.synapses_weights.append(random.random())
            else:
                self.synapses_weights.append(weight_list[w])

    def _calculate_state(self):
        self.axon_signal = np.dot(self.synapses_weights, self.synapse_signals) + self.bias
        self._calculated = True

    def set_synapse_signal(self, synapse_index, value):
        self.synapse_signals[synapse_index] = value
        self._calculated = False

    def get_axon_signal(self):
        if not self._calculated:
            self._calculate_state()
        return self.axon_signal


class NeuralNetworkLayer:
    def __init__(self, input_signal_num, neuron_num):
        self.neurons = []
        self.output_signals = [int]
        self._calculated = False

        for i in range(0, neuron_num):
            self.neurons.append(Neuron(input_signal_num))

    def set_signal(self, signal_index, value):
        for neuron in self.neurons:
            neuron.set_synapse_signal(signal_index, value)
        self._calculated = False

    def get_output_signals(self):
        if not self._calculated:
            self.output_signals = []
            for neuron in self.neurons:
                self.output_signals.append(neuron.get_axon_signal())
            self._calculated = True
        return self.output_signals



nnl = NeuralNetworkLayer(5, 3)

print(nnl.get_output_signals())
nnl.set_signal(1, 0)
print(nnl.get_output_signals())
