import numpy as np
from .activations import sigmoid, relu, sigmoid_derivative, relu_derivative, softmax, softmax_derivative, tanh, tanh_derivative
from scipy import signal, fft
from .loss import mse, mse_derivative, binary_cross_entropy, binary_cross_entropy_derivative, cross_entropy, cross_entropy_derivative

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
    
    
# when loss computation is only needed at end.
class RNNCell:
    def __init__(self, input_size, hidden_size, output_size, final_activation = None):
        self.input_size = input_size
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        # weights_xh[i][j] = weight from input i to hidden j
        self.weights_xh = np.random.randn(input_size, hidden_size)
        
        # weights_hh[i][j] = weight from hidden i to hidden j
        self.weights_hh = np.random.randn(hidden_size, hidden_size)
        
        # weights_hy[i][j] = weight from hidden i to output j
        self.weights_hy = np.random.randn(hidden_size, self.output_size) # output size might be differnt than input
        self.input = None
        self.hidden_state = None
        self.output = None
        self.hy_grad = np.zeros_like(self.weights_hy)
        self.hh_grad = np.zeros_like(self.weights_hh)
        self.xh_grad = np.zeros_like(self.weights_xh)
        self.history = []
        
        if final_activation != None:
            self.final_activation = final_activation
        if self.final_activation == sigmoid:
            self.final_activation_derivative = sigmoid_derivative
        elif self.final_activation == tanh:
            self.final_activation_derivative = tanh_derivative
        elif self.final_activation == relu:
            self.final_activation_derivative = relu_derivative
        elif self.final_activation == softmax:
            self.final_activation_derivative = softmax_derivative
  
            
        
    def forward_one(self,input, target, hidden_state = None):
        if hidden_state is None:
            hidden_state = np.zeros((1, self.hidden_size))
        self.hidden_state = hidden_state
        w1 = input @ self.weights_xh
        w2 = hidden_state @ self.weights_hh
        unactivated_hidden_state = w1   + w2
        new_hidden_state = tanh(unactivated_hidden_state)
        unactivated_output = new_hidden_state @ self.weights_hy
        if self.final_activation != None:
            activated_output = self.final_activation(unactivated_output)
        else:    
            activated_output = unactivated_output
            
        self.history.append((input, hidden_state, unactivated_hidden_state, unactivated_output,activated_output,new_hidden_state, target))
        return activated_output, new_hidden_state
    
    def forward(self, inputs, targets):
        self.inputs = inputs
        for i in range(len(inputs)):
            self.output, self.hidden_stafte = self.forward_one(inputs[i],targets[i] ,self.hidden_state)
            self.output = np.clip(self.output, 1e-10, 1- 1e-10)
        return self.output

    def backward(self,lr, loss):
        self.reset_grads()
        output_grad = np.zeros((1, self.hidden_size))
        
        if loss == 'binary':
            loss = softmax
            loss_deri = softmax_derivative
        elif loss == 'cross_entropy':
            loss = cross_entropy
            loss_deri = cross_entropy_derivative
        elif loss == 'mse':
            loss = mse
            loss_deri = mse_derivative
        else:
            raise ValueError("Loss not supported")
        
    
        # output_grad = output_grad @ self.weights_hy.T
        tot_loss = 0
        while len(self.history) > 0:
            output_grad,loss_f = self.singl_back_prop(output_grad, loss, loss_deri)
            tot_loss += loss_f
            self.history.pop()
        
        self.weights_hh += self.hh_grad * lr
        self.weights_xh += self.xh_grad * lr
        self.weights_hy += self.hy_grad * lr

        return tot_loss
        
    def singl_back_prop(self, output_grad, loss_fn,loss_derivative_fn):
        input, hidden_state, unactivated_hidden_state, unactivated_output, activated_output,new_hidden, target = self.history[-1]
        output_grad = output_grad
        
        loss = loss_fn(target, activated_output)
        output_grad1 = loss_derivative_fn(target, activated_output)
        if self.final_activation != None:
            output_grad1 = output_grad1 * self.final_activation_derivative(unactivated_output)
        self.hy_grad += new_hidden.T @output_grad1
        output_grad += output_grad1 @ self.weights_hy.T
        output_grad  = output_grad * tanh_derivative(unactivated_hidden_state)
        
        # self.xh_grad[i][j] = input[i] * output_grad[j]
        input = np.array([input])
        self.xh_grad  += input.T @output_grad
        self.hh_grad  += hidden_state.T @ output_grad
        
        # entire first column  of weights_hh is used to compute hidden_state[0], so we need to backpropagate through all of them
        # so the first column of weights_hh is transposed
        output_grad = output_grad @ self.weights_hh.T
        return output_grad, loss
    
    def reset_grads(self):
        self.hy_grad = np.zeros_like(self.weights_hy)
        self.hh_grad = np.zeros_like(self.weights_hh)
        self.xh_grad = np.zeros_like(self.weights_xh)
        
