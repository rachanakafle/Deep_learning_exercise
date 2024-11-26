
import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self,prediction_tensor, label_tensor):
        self.prediction_tensor= prediction_tensor
        # Add epsilon to avoid log(0) and ensure numerical stability
        epsilon = np.finfo(float).eps
        #  negative log-likelihood loss
        loss = -np.sum(label_tensor * np.log(prediction_tensor + epsilon))
        return loss


    def backward(self,label_tensor):
        # Add epsilon to avoid log(0) and ensure numerical stability
        epsilon = np.finfo(float).eps
        error_tensor =  -label_tensor / (self.prediction_tensor + epsilon)
        return error_tensor

