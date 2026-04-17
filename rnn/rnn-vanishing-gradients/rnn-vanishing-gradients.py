import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    largest_singular_value = np.linalg.norm(W_hh, ord=2)
    grad_norms = []
    for t in range(T):
        grad_norms.append(largest_singular_value ** t)
    return grad_norms
    pass