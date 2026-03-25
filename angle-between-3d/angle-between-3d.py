import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v = np.asarray(v)
    w = np.asarray(w)
    num = np.dot(v, w)
    v_norm = np.linalg.norm(v, ord=2)
    w_norm = np.linalg.norm(w, ord=2)
    if v_norm < 1e-10 or w_norm < 1e-10:
        return np.nan
    cos_value = num / (v_norm * w_norm) 
    return np.arccos(np.clip(cos_value, -1, 1))
    pass