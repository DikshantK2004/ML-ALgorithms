import numpy as np
import matplotlib.pyplot as plt
class LinearRegression():
    def __init__(self):
        self.w = None
        self.num_samples = None
        self.num_params = None
        
    def fit(self, X, y,epochs = 50, lr = 0.001, batch_size = 1):
            
        self.num_samples = X.shape[0]
        self.num_params = X.shape[1] + 1
        self.w = np.random.random(self.num_params)
        self.train(X, y, epochs, lr, batch_size)
    
    def predict(self, X):
        try:
            X_new = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        except np.AxisError:
            raise Exception("X must be a 2D array")
        
        return np.dot(X_new, self.w)
    
    def train(self, X, y,epochs = 50, lr = 0.001, batch_size = 1):
        """
            By default it is sort of stochastic gradient descent, but you can change it to batch gradient descent by setting batch_size to the number of samples
            mini-batch gradient descent by setting batch_size to a number less than the number of samples
        """
        if batch_size > self.num_samples:
            raise Exception("Batch size must be less than the number of samples")
        for epoch in range(epochs):
            for i in range(0, self.num_samples, batch_size):
                if i + batch_size > self.num_samples:
                    X_batch = X[i:]
                    y_batch = y[i:]
                else:
                    X_batch = X[i:i+batch_size]
                    y_batch = y[i:i+batch_size]
                
                y_pred = self.predict(X_batch)
                print(y_pred)
                error = y_batch - y_pred
                gradient = np.dot(X_batch.T, error)
                self.w += lr * gradient
                
            print(f"Epoch: {epoch}, Loss: {np.mean(error)}")
        
        
if __name__=="__main__":
    X = np.array([1,2,3,4,5,6])
    Y = np.array([2,3,4,5,6,7])
    
    plt.plot(X, Y, 'ro')
    
        
        