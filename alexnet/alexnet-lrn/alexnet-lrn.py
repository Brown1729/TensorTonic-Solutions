import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """Apply Local Response Normalization across channels."""
    # YOUR CODE HERE

    batch_size, channel_size, heigth, width = x.shape
    if n % 2 == 0:
        n += 1
    
    pad_size = n // 2
    padded_square = np.pad(x**2, ((0, 0), (pad_size, pad_size), (0, 0), (0, 0)), mode='constant')

    sum_square = np.zeros_like(x)
    for i in range(channel_size):
        sum_square[:, i, :, :] = np.sum(padded_square[:, i:i+n, :, :], axis=1)

    return x / ((k + alpha * sum_square) ** beta)
    
    pass