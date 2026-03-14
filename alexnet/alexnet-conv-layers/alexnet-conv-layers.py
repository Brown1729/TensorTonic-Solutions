import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    # YOUR CODE HERE
    batch_size = image.shape[0]
    height = image.shape[1]
    width = image.shape[2]
    channel = image.shape[3]
    padding = 2
    stride = 4
    kernel_size = 11
    filter_size = 96
    return np.zeros([batch_size, int((height + 2 * padding - kernel_size) / stride) + 1, 
                     int((width + 2 * padding - kernel_size) / stride) + 1, filter_size])
    pass