def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    Y = [[0 for i in range(len(W[0]))] for i in range(len(X))]
    for i in range(len(X)):
        for j in range(len(W[0])):
            Y[i][j] = sum(X[i][k] * W[k][j] for k in range(len(X[0]))) + b[j]                
    return Y