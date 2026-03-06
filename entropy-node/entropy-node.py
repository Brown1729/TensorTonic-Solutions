import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    vals, indices = np.unique(np.asarray(y), return_counts=True)
    freqs = indices / len(y)
    return - np.dot(freqs, np.log2(freqs))
    pass