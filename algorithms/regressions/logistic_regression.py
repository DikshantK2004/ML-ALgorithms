import numpy as np


class LogisticRegression():
    def __init__(self):
        self.w = None
        self.num_samples = None
        self.num_params = None
        self.losses = None
        self.lr_start = 0.01
        self.lr_end = 0.0000001
    
    def fit(self, X, y, epochs=50,  batch_size=1):
        self.num_samples = X.shape[0]
        self.num_params = X.shape[1] + 1
        self.w = np.random.random(self.num_params)
        self.train(X, y, epochs, batch_size)
    
    def predict(self, X):
        try:
            X_new = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        except np.AxisError:
            raise Exception("X must be a 2D array")
        
        return self.sigmoid(np.dot(X_new, self.w))

    def predict_without_expansion(self, X):
        return self.sigmoid(np.dot(X, self.w))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def train(self, X, y, epochs,  batch_size) :
        """
        By default it is sort of stochastic gradient descent, but you can change it to batch gradient descent by setting batch_size to the number of samples
        mini-batch gradient descent by setting batch_size to a number less than the number of samples
        """
        
        
        loss_history = np.array([])
        X_new = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        if batch_size > self.num_samples:
            raise Exception(
                "Batch size must be less than the number of samples")
        for epoch in range(epochs):
            lr = self.lr_start - (self.lr_start - self.lr_end) * epoch / epochs
            h_values = np.array([])
            for i in range(0, self.num_samples, batch_size):
                if i + batch_size > self.num_samples:
                    X_batch = X_new[i:]
                    y_batch = y[i:]
                else:
                    X_batch = X_new[i:i+batch_size]
                    y_batch = y[i:i+batch_size]
                
                y_pred = self.predict_without_expansion(X_batch)
                y_pred = np.clip(y_pred, 1e-10, 1-1e-10)
                error = y_batch - y_pred
                
                h_values = np.concatenate([h_values, y_pred], axis=None)
                gradient = np.dot(X_batch.T, error)
                self.w += lr * gradient
            loss = -np.sum(y * np.log(h_values) + (1 - y) * np.log(1 - h_values))
            loss_history = np.concatenate([loss_history, loss], axis=None)
            print(f"Epoch: {epoch}, Loss: {loss}")
        
        self.losses = loss_history
        
if __name__=="__main__":
    len = 10
    X_points = np.arange(1, len + 1)
    X = np.reshape(X_points, (X_points.shape[0], 1))
    y = X_points >5
    lr = LogisticRegression()
    lr.fit(X, y, epochs=100000, lr=0.0001, batch_size=1)
    print(lr.predict(X) > 0.5)
        