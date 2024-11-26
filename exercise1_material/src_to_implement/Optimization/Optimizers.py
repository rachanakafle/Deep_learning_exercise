class Sgd:
    def __init__(self,learning_rate:float):
        self.learning_rate=learning_rate


    def calculate_update(self,weight_tensor,gradient_tensor):
        #Wt + 1 = Wt − η · En upadte weight using gradient
        update_weight= weight_tensor-self.learning_rate*gradient_tensor
        return update_weight