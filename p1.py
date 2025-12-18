import numpy as np
import nnfs
from test_data_generator import spiral_data

nnfs.init()

# np.random.seed(0)

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


class ActivationSoftmax:
    def forward(self, inputs):
        # axis = 1 does row-wise operation, keepdims=True maintains dimensions
        # it enables broadcasting to happen and applies the row-wise operation
        # to each element in a specific row
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def forward(self, output, y):
        pass


class CategoricalCrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # need to clip to handle infinity errors with log
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Truth labels (y_true) can look like:
        # (i) [1, 0, 2] -> scalar values
        # (ii) [[0, 1, 0], [1, 0, 0], [0, 0, 1]] -> one-hot encoded
        # Hence, need to handle both cases
        correct_confidences = np.sum(y_pred_clipped * 0, axis=1)
        if len(y_true.shape) == 1:
            # for each k iterated in y_true, we only want the probability for
            # each sample[k] in samples
            # everything else gets multiplied by 0 in the cross-entropy loss formula,
            # so it is irrelevant to the loss calculation
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            print("Truth label format not supported")

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = spiral_data(100, 3)
dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = CategoricalCrossEntropyLoss()
loss = loss_function.calculate(activation2.output, y)
print(loss)


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
