import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    g_t = np.asarray(grad)
    m_t = np.asarray(m)
    v_t = np.asarray(v)
    param_t = np.asarray(param)
    m_new = beta1 * m_t + (1 - beta1) * g_t
    v_new = beta2 * v_t + (1 - beta2) * g_t * g_t
    m_correct = m_new / (1 - beta1 ** t)
    v_correct = v_new / (1 - beta2 ** t)
    param_new = param_t - lr * m_correct / (np.sqrt(v_correct) + eps) 
    return (param_new, m_new, v_new)
    pass