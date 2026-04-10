import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    
    Args:
        X: Input data of shape (batch_size, T, input_dim)
        h_0: Initial hidden state of shape (batch_size, hidden_dim)
        W_xh: Input-to-hidden weights of shape (hidden_dim, hidden_dim)
        W_hh: Hidden-to-hidden weights of shape (hidden_dim, hidden_dim)
        b_h: Bias of shape (hidden_dim,)
    
    Returns:
        Tuple (h_all, h_T) where:
            h_all: All hidden states, shape (batch_size, T, hidden_dim)
            h_T: Final hidden state, shape (batch_size, hidden_dim)
    """
    # YOUR CODE HERE
    batch_size, T, input_dim = X.shape
    hidden_dim = h_0.shape[1]
    h_t = np.zeros((batch_size, hidden_dim))
    h_all = []
    for i in range(T):
        x_t = X[:, i, :]
        h_t = np.tanh(h_t @ W_hh.T +  x_t @ W_xh.T + b_h)
        h_all.append(h_t)
    return np.stack(h_all, axis=1), h_t