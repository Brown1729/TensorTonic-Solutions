import numpy as np

def relu(x):
    return np.maximum(0, x)

class ConvBlock:
    """
    Convolutional Block with projection shortcut.
    Used when input/output dimensions differ.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path weights
        self.W1 = np.random.randn(in_channels, out_channels) * 0.01
        self.W2 = np.random.randn(out_channels, out_channels) * 0.01
        
        # Shortcut projection (1x1 conv equivalent)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with projection shortcut.
        """
         # Main path: linear -> ReLU -> linear
        # 反过来
        out = x @ self.W1          # (batch, in_channels) -> (batch, out_channels)
        out = relu(out)
        out = out @ self.W2        # (batch, out_channels) -> (batch, out_channels)
        
        # Shortcut path: projection
        shortcut = x @ self.Ws     # (batch, in_channels) -> (batch, out_channels)
        
        # Summation
        return relu(out + shortcut)
