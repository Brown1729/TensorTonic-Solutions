import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    # YOUR CODE HERE
    # x = x.reshape(1, -1)
    gradients_F = np.asarray(gradients_F)
    gradients_F = np.identity(gradients_F[0].shape[-1]) + gradients_F
    if gradients_F.shape[0] == 1:
        return x @ gradients_F[0]
    else:
        return x @ np.linalg.multi_dot(gradients_F)
    pass

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # YOUR CODE HERE
    gradients_F = np.asarray(gradients_F)
    if gradients_F.shape[0] == 1:
        return x @ gradients_F[0]
    else:
        return x @ np.linalg.multi_dot(gradients_F)
    pass
