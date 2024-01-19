import copy
import math
import numpy as np

class Layer:
    def __init__(self, node, num):
        # initialize the layer by copying num amount of given nodes and initializing the weights and biases
        self.nodes = [copy.deepcopy(node) for i in range(num)]
        self.activation_function = None  # Set the activation function
        self.initializeNodesXavier(self.nodes)

    def forward_propagation(self, inputs):
        # Compute the weighted sum of inputs, apply the activation function, and return the output
        # ...
        pass
