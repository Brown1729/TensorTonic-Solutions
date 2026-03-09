import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here

    Q_full = Q @ W_q
    K_full = K @ W_k
    V_full = V @ W_v
    Q_heads = Q_full.reshape(Q_full.shape[0], Q_full.shape[1], num_heads, -1).transpose(0, 2, 1, 3)
    V_heads = V_full.reshape(V_full.shape[0], V_full.shape[1], num_heads, -1).transpose(0, 2, 1, 3)
    K_heads = K_full.reshape(K_full.shape[0], K_full.shape[1], num_heads, -1).transpose(0, 2, 1, 3)
    scores = scaled_dot_product_attention(Q_heads, K_heads, V_heads)
    return scores.transpose(0, 2, 1, 3).reshape(Q_full.shape[0], Q_full.shape[1], Q_full.shape[2]) @ W_o


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 mask: np.ndarray = None) -> np.ndarray:
    """
    Compute scaled dot-product attention.
    """
    return softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(K.shape[-1])) @ V