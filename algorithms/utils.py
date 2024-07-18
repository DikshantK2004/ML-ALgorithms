import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def plot_various_lines(data: List[Tuple[np.ndarray, np.ndarray]], types = List[str], legend = List[str]):
    for i, (X, Y) in enumerate(data):
        plt.plot(X, Y, types[i])
    plt.legend(legend)
    plt.plot()