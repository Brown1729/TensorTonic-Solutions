import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    w_ = np.asarray(w)
    g_ = np.asarray(g)
    s_ = np.asarray(s)
    s_ = beta * s_ + (1 - beta) * g_ * g_
    w_ = w_ - lr * g_ / np.sqrt(s_ + eps)
    return (w_, s_)
    pass