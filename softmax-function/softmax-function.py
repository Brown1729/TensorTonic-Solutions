import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x)
    max_x = np.max(x, axis=-1, keepdims=True)
    e_demax = np.exp(x - max_x)
    denom = np.sum(e_demax, axis=-1, keepdims=True)
    return e_demax / denom
    pass