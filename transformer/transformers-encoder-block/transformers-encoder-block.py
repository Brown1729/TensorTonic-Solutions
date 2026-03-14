import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    x = np.asarray(x)
    mu = np.mean(x, axis=-1, keepdims=True)
    sigma_2 = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(sigma_2 + eps) + beta
    pass

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    Q_full = Q @ W_q
    K_full = K @ W_k
    V_full = V @ W_v
    # bath_size num_heads num_features d_k
    Q_heads = Q_full.reshape(Q_full.shape[0], Q_full.shape[1], num_heads, -1).transpose(0, 2, 1, 3)
    V_heads = V_full.reshape(V_full.shape[0], V_full.shape[1], num_heads, -1).transpose(0, 2, 1, 3)
    K_heads = K_full.reshape(K_full.shape[0], K_full.shape[1], num_heads, -1).transpose(0, 2, 1, 3)
    scores = scaled_dot_product_attention(Q_heads, K_heads, V_heads)
    # bathc_size num_features d_model
    return scores.transpose(0, 2, 1, 3).reshape(Q_full.shape[0], Q_full.shape[1], Q_full.shape[2]) @ W_o
    pass

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 mask: np.ndarray = None) -> np.ndarray:
    """
    Compute scaled dot-product attention.
    """
    return softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(K.shape[-1])) @ V

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    return np.dot(np.maximum(0, np.dot(x, W1) + b1), W2) + b2
    pass

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    x_ = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)
    return layer_norm(x_ + feed_forward(x_, W1, b1, W2, b2), gamma2, beta2)
    pass