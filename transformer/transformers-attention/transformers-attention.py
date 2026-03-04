import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    # 因为QK^T的结果为(batch_size, q_pose, k_pose) 要对
    return torch.matmul(F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1)), dim=-1), V)
    pass