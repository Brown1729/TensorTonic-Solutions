import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x_ = np.asarray(x)
    return (np.exp(x_) - np.exp(-x_)) / (np.exp(x_) + np.exp(-x_))
    pass