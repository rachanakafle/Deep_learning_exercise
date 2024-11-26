import numpy as np
from .Base import *

class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        # store input tensor for backward pass
        self.input_tensor= input_tensor
        #f(x) = max(0, x)
        return np.maximum(0,input_tensor)

    def backward(self, error_tensor):
        error_tensor[self.input_tensor <= 0] = 0
        return error_tensor
