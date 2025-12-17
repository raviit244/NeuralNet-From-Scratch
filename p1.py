import numpy as np
import nnfs
from test_data_generator import spiral_data

nnfs.init()

np.random.seed(0)

# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

class LayerDense:
    def __init__(self, features, datapoints):
        # Initialise with shape: (features x datapoints)
        # No need to use transpose for batch forward pass anymore
        self.weights = 0.1 * np.random.randn(features, datapoints)
        self.biases = np.zeros((1, datapoints))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = LayerDense(4, 5)
layer2 = LayerDense(5, 2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)


# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
#
# biases = [2, 3, 0.5]
#
# weights2 = [[0.1, -0.14, 0.5],
#             [-0.5, 0.12, -0.33],
#             [-0.44, 0.73, -0.13]]
#
# biases2 = [-1, 2, -0.5]
#
# # layer1_outputs = np.dot(inputs, np.array(weights).T) + np.dot(np.ones((3, 1)), np.array(biases).reshape(1, 3))
# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
#
# print(layer2_outputs)


# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0
#     for neuron_input, weight in zip(inputs, neuron_weights):
#         neuron_output = neuron_input * weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)

# print(layer_outputs)
