import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # YOUR CODE HERE
        h_t = h_0
        batch_size, T, input_dim = X.shape
        input_dim, hidden_dim = h_t.shape
        output_dim = self.output_dim
        y_seq = np.zeros((batch_size, T, output_dim))
        for t in range(T):
            h_t = np.tanh(X[:, t, :] @ self.W_xh.T + h_t @ self.W_hh.T + self.b_h)
            y_seq[:, t, :] = (h_t @ self.W_hy.T + self.b_y)

        h_final = h_t
        return (y_seq, h_final)
        pass