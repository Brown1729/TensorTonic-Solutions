import numpy as np


def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    vec_seq = np.arange(seq_length).reshape(-1, 1)
    vec_d_model = np.arange(d_model).reshape(1, -1)
    vec_d_model = np.exp(-np.log(10000) * 2 * (vec_d_model // 2) / d_model)
    w = np.dot(vec_seq, vec_d_model)
    for i in range(w.shape[1]):
        if i % 2 == 0:
            w[:, i] = np.sin(w[:, i])
        else:
            w[:, i] = np.cos(w[:, i])
    return w
    pass