import numpy as np
from .activations import sigmoid, relu, sigmoid_derivative, relu_derivative, softmax, softmax_derivative
from scipy import signal, fft


# accepts as batches only, so from flatten send (1, something)
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
    
    def reset(self):
        self.weights = np.random.randn(self.input_size, self.output_size)
        self.biases = np.random.randn(self.output_size)
        self.input = None
        self.output = None
    
class Relu:
    def __init__(self):
        self.input = None
        self.output = None
        self.activation = relu
        self.activation_derivative = relu_derivative
    
    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output
    
    def backward(self, output_grad, lr):
        return output_grad * self.activation_derivative(self.input)
    
    def __repr__(self):
        return 'Relu'
    
    def __str__(self):
        return 'Relu'
    
    def reset(self):
        self.input = None
        self.output = None
        
class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None
        self.activation = sigmoid
        self.activation_derivative = sigmoid_derivative
    
    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output
    
    def backward(self, output_grad, lr):
        return output_grad * self.activation_derivative(self.input)
    
    def __repr__(self):
        return 'Sigmoid'
    
    def __str__(self):
        return 'Sigmoid'
    
    def reset(self):
        self.input = None
        self.output = None
        
class Flatten:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        self.input = input
        prod = 1
        for i in input.shape[:]:
            prod *= i
        self.output = input.reshape(1, prod)
        return self.output
    
    def backward(self, output_grad, lr):
        return output_grad.reshape(self.input.shape)
    
    def __repr__(self):
        return 'Flatten'
    
    def __str__(self):
        return 'Flatten'
    
    def reset(self):
        self.input = None
        self.output = None
        


class Conv2D:
    def __init__(self, n_channels, num_filter, kernel_size=(3, 3), stride=1, padding=0):
        self.n_channels = n_channels
        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(num_filter, n_channels, *kernel_size)
        self.biases = np.random.randn(num_filter)
        self.input = None
        self.output = None
        self.input_padded = None

    def pad_input(self, input):
        channels, row, cols = input.shape
        input_padded = np.zeros((channels, row + 2*self.padding, cols + 2*self.padding))
        input_padded[:, self.padding:row + self.padding, self.padding:cols + self.padding] = input
        return input_padded

    def compute_output_shape(self, input_shape):
        channels, row, cols = input_shape
        out_height = int((row + 2*self.padding - self.kernel_size[0]) / self.stride + 1)
        out_width = int((cols + 2*self.padding - self.kernel_size[1]) / self.stride + 1)
        return self.num_filter, out_height, out_width

    def forward(self, input):
        self.input = input
        input_padded = self.pad_input(input)
        self.input_padded = input_padded
        channels, row, cols = input.shape
        output_shape = self.compute_output_shape(input.shape)
        self.output = np.zeros(output_shape)

        for i in range(0, row + 2*self.padding - self.kernel_size[0] + 1, self.stride):
            for j in range(0, cols + 2*self.padding - self.kernel_size[1] + 1, self.stride):
                i_end = i + self.kernel_size[0]
                j_end = j + self.kernel_size[1]
                input_slice = input_padded[:, i:i_end, j:j_end]
                self.output[:, int(i/self.stride), int(j/self.stride)] = (
                    np.tensordot(input_slice, self.weights, axes=((0, 1, 2), (1, 2, 3))) + self.biases
                )

        return self.output

    def backward(self, output_grad, lr):
        _, out_height, out_width = output_grad.shape
        grad_w = np.zeros_like(self.weights)
        grad_b = np.zeros_like(self.biases)
        grad_input = np.zeros_like(self.input_padded)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride
                w_end = w_start + self.kernel_size[1]

                input_slice = self.input_padded[:, h_start:h_end, w_start:w_end]
                for f in range(self.num_filter):
                    grad_input[:, h_start:h_end, w_start:w_end] += (
                        self.weights[f] * output_grad[f, i, j]
                    )
                    grad_w[f] += input_slice * output_grad[f, i, j]
                    grad_b[f] += output_grad[f, i, j]

        if self.padding != 0:
            grad_input = grad_input[:, self.padding:-self.padding, self.padding:-self.padding]

        self.weights -= lr * grad_w
        self.biases -= lr * grad_b

        return grad_input

    def reset(self):
        self.weights = np.random.randn(self.num_filter, self.n_channels, *self.kernel_size)
        self.biases = np.random.randn(self.num_filter)
        self.input = None
        self.output = None
        self.input_padded = None

    def __repr__(self):
        return f'Conv2D(num_filter={self.num_filter}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'

    def __str__(self):
        return f'Conv2D(num_filter={self.num_filter}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'
        
        
        
class MaxPool:
    def __init__(self, pool_size=(2, 2), stride=1, padding=0):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.input = None
        self.output = None
        self.input_padded = None

    def pad_input(self, input):
        channels, row, cols = input.shape
        input_padded = np.zeros((channels, row + 2*self.padding, cols + 2*self.padding))
        input_padded[:, self.padding:row + self.padding, self.padding:cols + self.padding] = input
        return input_padded

    def compute_output_shape(self, input_shape):
        channels, row, cols = input_shape
        out_height = int((row + 2*self.padding - self.pool_size[0]) / self.stride + 1)
        out_width = int((cols + 2*self.padding - self.pool_size[1]) / self.stride + 1)
        return channels, out_height, out_width

    def forward(self, input):
        self.input = input
        input_padded = self.pad_input(input)
        self.input_padded = input_padded
        channels, row, cols = input.shape
        output_shape = self.compute_output_shape(input.shape)
        self.output = np.zeros(output_shape)

        for i in range(0, row + 2*self.padding - self.pool_size[0] + 1, self.stride):
            for j in range(0, cols + 2*self.padding - self.pool_size[1] + 1, self.stride):
                i_end = i + self.pool_size[0]
                j_end = j + self.pool_size[1]
                input_slice = input_padded[:, i:i_end, j:j_end]
                self.output[:, int(i/self.stride), int(j/self.stride)] = np.max(input_slice, axis=(1, 2))

        return self.output

    def backward(self, output_grad, lr=None):
        _, out_height, out_width = output_grad.shape
        grad_input = np.zeros_like(self.input_padded)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride
                w_end = w_start + self.pool_size[1]

                input_slice = self.input_padded[:, h_start:h_end, w_start:w_end]
                max_vals = np.max(input_slice, axis=(1, 2), keepdims=True)
                mask = (input_slice == max_vals)
                grad_input[:, h_start:h_end, w_start:w_end] += mask * output_grad[:, i, j].reshape(-1, 1, 1)

        if self.padding != 0:
            grad_input = grad_input[:, self.padding:-self.padding, self.padding:-self.padding]

        return grad_input

    def reset(self):
        self.input = None
        self.output = None
        self.input_padded = None

    def __repr__(self):
        return f'MaxPool(pool_size={self.pool_size}, stride={self.stride}, padding={self.padding})'

    def __str__(self):
        return f'MaxPool(pool_size={self.pool_size}, stride={self.stride}, padding={self.padding})'

class SoftMax:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = softmax(input)
        return self.output

    def backward(self, output_grad, lr):
        return output_grad * softmax_derivative(self.input)
    
    def reset(self):
        self.input = None
        self.output = None

    def __repr__(self):
        return 'SoftMax'

    def __str__(self):
        return 'SoftMax'