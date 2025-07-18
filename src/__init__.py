"""Trading RL Agent - A reinforcement learning framework for algorithmic trading."""

import numpy as np

# Alias for compatibility with code expecting np.float_
try:
    if not hasattr(np, "float_"):
        np.float64 = np.float64
except AttributeError:
    # np.float_ was removed in newer numpy versions
    pass
