import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x_ = np.sort(np.asarray(x))
    return np.percentile(x_, q, method='linear')
    pass