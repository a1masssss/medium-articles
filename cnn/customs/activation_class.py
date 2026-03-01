from customs.layer_class import Layer
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    def backward(self, output_error, lr):
        # chain rule 
        return output_error * self.activation_prime(self.input)


