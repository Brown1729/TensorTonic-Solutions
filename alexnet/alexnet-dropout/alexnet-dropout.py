import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """Apply dropout to input."""
    # YOUR CODE HERE
    if training:
        return 1 / (1 - p) * x * np.random.binomial(1, 1 - p, x.shape)
    else:
        return x
    pass