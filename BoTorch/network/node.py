class Node:
    weight = 0
    bias = 0
    
    def __init__(self, activation):
        self.activation = activation
        # Additional initializations

    def forward(self, x):
        return self.activation.forward(x)

    def backward(self, x):
        return self.activation.backward(x)
    
    