import numpy as np
from customs.activation_class import Activation
class Softmax(Activation):
    def __init__(self):
        def softmax(x):
            exps =  np.exp(x - np.max(x, axis=-1, keepdims=True)) # axis=-1 calculates probability horizontaly, keepdim is for broadcasting
            return exps / np.sum(exps, axis=-1, keepdims=True) # getting [0, 1]
        def softmax_prime(x):
            return 1
        
        super().__init__(softmax, softmax_prime)
        