
import numpy as np
from .Base import *


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True

        # Initialize weights with shape (input_size + 1, output_size) to include bias
        self.weights = np.random.uniform(0, 1, size=(input_size + 1, output_size))

        self._optimizer = None
        self._gradient_tensor = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, setter):
        self._optimizer = setter

    def forward(self, input_tensor):

        # Add a bias column of ones to the input tensor
        self.input_tensor = np.hstack([input_tensor, np.ones((input_tensor.shape[0], 1))])

        # Compute output as y = WX
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        # Compute gradient of weights (including bias)
        self._gradient_tensor = np.dot(self.input_tensor.T, error_tensor)

        # Compute error tensor for the previous layer (exclude the bias row in weights)
        error_tensor_prev = np.dot(error_tensor, self.weights[:-1, :].T)

        # Update weights if optimizer is available
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_tensor)

        return error_tensor_prev

    @property
    def gradient_weights(self):
        return self._gradient_tensor


