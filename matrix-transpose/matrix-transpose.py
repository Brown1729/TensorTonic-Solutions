import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    return np.array([[ A[row][col] for row, _ in enumerate(A) ] for col, _ in enumerate(A[0])])
    pass
