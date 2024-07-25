import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
def plot_various_lines(data: List[Tuple[np.ndarray, np.ndarray]], types = List[str], legend = List[str]):
    for i, (X, Y) in enumerate(data):
        plt.plot(X, Y, types[i])
    plt.legend(legend)
    plt.plot()
    
def plot_loss_history(loss_history: List[float]):
    plt.plot( [i for i in range(len(loss_history))], loss_history, 'r-')
    
def compute_loss(y_pred, y_true, loss):
    return loss(y_true, y_pred)

def compute_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def plot_decision_boundary(model, X :np.ndarray, Y:np.ndarray):
    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    
    X_to_pred_on = np.column_stack((xx.ravel(), yy.ravel()))
    
    y_logits = model.predict(X_to_pred_on)
    
    if len(np.unique(Y)) > 2:
        y_pred = np.argmax(y_logits, axis = 1)
    else:
        y_pred = np.round(y_logits)
        
    y_pred = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, cmap =plt.cm.RdYlBu ,alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot_decision_boundary_on_axes(model, X: np.ndarray, Y: np.ndarray, ax: plt.Axes):
    """
    Plot decision boundary of a model on specific axes.

    Parameters:
    model: The trained model with a predict method.
    X: np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    Y: np.ndarray, shape (n_samples,)
        Labels.
    ax: plt.Axes
        The axes on which to plot.
    """
    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    
    X_to_pred_on = np.column_stack((xx.ravel(), yy.ravel()))
    
    y_logits = model.predict(X_to_pred_on)
    
    if len(np.unique(Y)) > 2:
        y_pred = np.argmax(y_logits, axis=1)
    else:
        y_pred = np.round(y_logits)
        
    y_pred = y_pred.reshape(xx.shape)
    ax.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.RdYlBu)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary')
