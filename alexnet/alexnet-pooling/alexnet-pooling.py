import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """Apply 2D max pooling (shape simulation)."""
    # YOUR CODE HERE
    batch_size, height, width, channel_size = x.shape
    pool_height = (height - kernel_size) // stride + 1
    pool_width = (width - kernel_size) // stride + 1
    return np.zeros((batch_size, pool_height, pool_width, channel_size))
    pass