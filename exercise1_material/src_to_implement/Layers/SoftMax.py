
import numpy as np
from .Base import *
class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        # Store the input tensor for backward pass
        self.input_tensor= input_tensor

        # Stabilize input by subtracting the max value from each row
        stabilized_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)

        # exponential for each element
        exp_values = np.exp(stabilized_input)

        # Normalize each row (divide by the sum of exponentials for each row)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # Store probabilities for use in backward pass
        self.output_tensor = probabilities
        return probabilities


    def backward(self,error_tensor):
        error_tensor = self.output_tensor * (error_tensor - np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True))
        return error_tensor


