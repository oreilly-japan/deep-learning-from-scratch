import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)
