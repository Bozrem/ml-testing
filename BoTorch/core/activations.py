import numpy as np

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0) * 1

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)