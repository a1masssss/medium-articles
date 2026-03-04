import numpy as np
from scipy import signal 
from customs.layer_class import Layer

class Convolutional(Layer):
    def __init__(self, input_shape: tuple[int, int, int], kernel_size: int, depth: int):
        input_depth, input_height, input_width  = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1) 
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * 0.1
        self.biases = np.zeros(self.output_shape)   


    def forward(self, input):
        self.input = input 
        self.output = np.copy(self.biases) # (depth, output_height, output_width)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], 'valid')
        return self.output

    def backward(self, output_grad, lr):
        kernels_grad = np.zeros(self.kernels_shape)
        input_grad = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_grad[i, j] = signal.correlate2d(self.input[j], output_grad[i], 'valid')
                input_grad[j] += signal.convolve2d(output_grad[i], self.kernels[i, j], 'full')

        self.kernels -= lr * kernels_grad
        self.biases -= lr * output_grad
        return input_grad
        