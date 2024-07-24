import numpy as np
from .activations import sigmoid, relu, sigmoid_derivative, relu_derivative
class Dense:
    def __init__(self, input_size, output_size, activation = None):
        self.input_size =  input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) # each column shows the weights to go to next output neuron
        self.biases = np.random.randn(output_size) # 1 * output_size
        self.activation = activation
        self.activation_derivative = None
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        
        self.input = None # m * input_size
        self.output = None # m * output_size
        ## Dimensions of Output Gradient:- m * output_size
        ## Formulae used:
        """
        weights -= input.T @ output_grad * lr , dimension of input.T @ output_grad = input_size * output_size
        biases -= np.sum(output_grad) * lr , dimension of np.sum(output_grad) = 1 * output_size
        """

    def forward(self, input):
        self.input = input
        self.output = input @ self.weights + self.biases
        if self.activation is not None:
            return self.activation(self.output).clip(min=1e-12, max=1-1e-12)
        return self.output

    def backward(self, output_grad, lr):
        if self.activation is not None:
            output_grad = output_grad * self.activation_derivative(self.output)
        self.weights -= self.input.T @ output_grad * lr
        self.biases -= np.sum(output_grad) * lr
        
        return output_grad @ self.weights.T
    
    def __repr__(self):
        return f'Layer({self.input_size}, {self.output_size}, {self.activation})'
    
    def __str__(self):
        return f'Layer({self.input_size}, {self.output_size}, {self.activation})'
    
        
        
        
        