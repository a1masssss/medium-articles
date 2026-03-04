from customs.layer_class import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))


    def forward(self, input):
        self.input = input 
        return np.dot(self.input, self.weights) + self.bias
    
    def backward(self, output_error, lr):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)


        self.weights -= lr * weights_error
        self.bias -= lr * output_error
        
        return input_error