import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224) -> np.ndarray:
    """Extract a random crop from the image."""
    # YOUR CODE HERE
    ori_height, ori_width = image.shape[0], image.shape[1]
    top = np.random.randint(0, ori_height - crop_size)
    left = np.random.randint(0, ori_height - crop_size)
    return image[top : top + crop_size, left : left + crop_size]
    pass

def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Randomly flip image horizontally."""
    # YOUR CODE HERE
    if np.random.rand(1) > p:
        return np.fliplr(image)
    return image
    pass