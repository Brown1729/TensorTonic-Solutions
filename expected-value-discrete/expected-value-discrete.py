import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    if not np.allclose(1, np.sum(p)):  # Strictly, allclose hits a tolerance at 10^-8
        raise ValueError("exception: probabilities must sum to 1")
    return np.dot(np.asarray(x), np.asarray(p))
    pass
