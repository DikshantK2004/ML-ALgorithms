import numpy as np

#! in general loss computation is better after epoch 
# https://stats.stackexchange.com/questions/436154/is-it-better-to-accumulate-accuracy-and-loss-during-an-epoch-or-recompute-all-of
class SoftmaxRegression():
    def __init__(self):
        # w is a matrix of shape (num_params, num_classes)
        self.w = None
        self.num_samples = None
        self.num_params = None
        self.losses = None
        self.num_classes = None
        self.lr_start = 0.01
        self.lr_end = 0.0000001
    
    def fit(self, X, y,num_classes ,epochs=50 ,batch_size=1, constant_lr = False, lr = 0.01):
        """
            X: 2D numpy array
            y: 1D numpy array
            num_classes: int
            all the values in y must be between 0 and num_classes - 1
        """
        self.num_samples = X.shape[0]
        self.num_params = X.shape[1] + 1
        # column vectors are the weights for each class
        self.w = np.random.random((self.num_params, num_classes))
        self.num_classes = num_classes
        self.train(X, y, epochs, batch_size, constant_lr, lr)
    
    def predict(self, X):
        try:
            X_new = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        except np.AxisError:
            raise Exception("X must be a 2D array")
        
        return self.softmax(X_new @ self.w)
    
    def predict_without_expansion(self, X):
        return self.softmax(np.dot(X, self.w))
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
    def train(self, X, y, epochs, batch_size, constant_lr, lr):
        self.losses = []
        X_new = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        
        Y_one_hot = np.zeros((self.num_samples, self.num_classes))
        Y_one_hot[np.arange(self.num_samples), y] = 1
        for epoch in range(epochs):
            losses_in_epoch = np.zeros(0)
            if not constant_lr:
                self.lr = self.lr_start - (epoch / epochs) * (self.lr_start - self.lr_end)
            else:
                self.lr = lr
            for i in range(0, self.num_samples, batch_size):
                if i + batch_size > self.num_samples:
                    batch_size = self.num_samples - i
                X_batch = X_new[i:i+batch_size]
                y_batch = Y_one_hot[i:i+batch_size]
                cur_loss = self.update(X_batch, y_batch)
                losses_in_epoch = np.append(losses_in_epoch, cur_loss)
            self.losses.append(np.mean(losses_in_epoch))
            
            print(f"Epoch {epoch+1}/{epochs} => Loss: {self.losses[-1]}")            
    
    def update(self, X, y):
        y_pred = self.predict_without_expansion(X) # shape: (batch_size, num_classes)
        # shape of y: (batch_size, num_classes)
        y_pred = np.clip(y_pred, 1e-10, 1-1e-10)
        error = y_pred- y # shape: (batch_size, num_classes)
        
        # shape of X: (batch_size, num_params)
        grads = X.T @ error # shape: (num_params, num_classes) which means one column for each class, shape of w
        # print(self.w.shape, grads.shape)
        self.w -= self.lr * grads
        # compute loss
        loss = -np.sum(y * np.log(y_pred)) 
    
        return loss
        
    def cross_entropy_loss(self, X, y):
        y_pred = self.predict_without_expansion(X)
        y_pred = np.clip(y_pred, 1e-10, 1-1e-10)
        Y_one_hot = np.zeros((self.num_samples, self.num_classes))
        Y_one_hot[np.arange(self.num_samples), y] = 1
        # dimensions of output loss: 
        return -np.sum(Y_one_hot * np.log(y_pred)) / self.num_samples
    
        
        
        

if __name__=="__main__":
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    model = SoftmaxRegression()
    model.fit(X, y, 3, epochs=50000, batch_size=X.shape[0]//10)
    
    