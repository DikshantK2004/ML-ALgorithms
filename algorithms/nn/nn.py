import numpy as np
from typing import List
from .loss import *
from .layers import *
# this class will be inherited by true models
class Sequential:
    def __init__(self, layers : List[Dense]):
        self.layers = layers
        self.loss = None
        self.loss_derivative = None
        self.output = None
        self.input = None
        self.lr_max = 0.1
        self.lr_min = 0.00000001
    
    def forward(self, input):
        self.input = input
        self.output = input
        for layer in self.layers:
            self.output = layer.forward(self.output)
            
        return self.output
    
    def backward(self, output_grad,lr):
        
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad, lr)
        return output_grad
    
    def fit(self, X, y, epochs, loss,  batch_size = 2):
        
        if batch_size > np.size(X):
            raise ValueError('Batch size should be less than or equal to the number of samples')
        if loss == 'mse':
            self.loss = mse
            self.loss_derivative = mse_derivative
        elif loss == 'cross_entropy':
            self.loss = cross_entropy
            self.loss_derivative = cross_entropy_derivative
        elif loss == 'binary_cross_entropy':
            self.loss = binary_cross_entropy
            self.loss_derivative = binary_cross_entropy_derivative
        else:
            raise ValueError('Invalid loss function')
            
        loss_history = []
        batched_data = np.array_split(X, len(X) // batch_size)
        batched_outs = np.array_split(y, len(y) // batch_size)
        for epoch in range(epochs):
            loss = 0
            lr = self.lr_min + (self.lr_max - self.lr_min) * np.exp(-epoch/epochs)
            for i,batch in enumerate(batched_data):
                batch_pred = self.forward(batch)
                batch_out = batched_outs[i]
                loss += self.loss(batch_out, batch_pred)
                loss_grad = self.loss_derivative(batch_out, batch_pred)
                self.backward(loss_grad, lr)
                
            print(f'Epoch: {epoch+1}, Loss: {loss}')
            loss_history.append(loss)
            
        return loss_history
            
if __name__=="__main__":
    from layers import *
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0],[ 1], [1], [0]])
    
    model = Sequential([
        Dense(2, 4, 'relu'),
        Dense(4, 1, 'sigmoid'),
    ])
    
    loss_history = model.fit(X, y, 10000,  'binary_cross_entropy')
    
    