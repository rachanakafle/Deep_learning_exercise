from functools import reduce

class NeuralNetwork:

    def __init__(self,optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
    def forward(self):
        # get input data and labels
        input_tensor, self.label_tensor = self.data_layer.next()

        # pass input_tensor through all layers using reduce
        input_tensor = reduce(lambda data, layer: layer.forward(data), self.layers, input_tensor)

        # compute loss or predictions using the loss layer
        self.prediction = self.loss_layer.forward(input_tensor, self.label_tensor)

        # Step Return the predictions or loss
        return self.prediction



    # def backward(self):


