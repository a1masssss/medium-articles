import numpy as np
from customs.activation_class import Activation
class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)
        def relu_prime(x):
            return x > 0
        super().__init__(relu, relu_prime)