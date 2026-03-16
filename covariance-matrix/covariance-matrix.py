import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if X.shape[0] < 2:
        return None
    if X.ndim != 2:
        return None
    mu = np.mean(X, axis=0)
    N = X.shape[0]
    X_centered = X - mu
    return 1 / (N - 1) * (X_centered.T @ X_centered)
    pass