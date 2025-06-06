# Monkey-patch numpy to restore deprecated attributes removed in NumPy 2.0
import numpy as np

# Restore np.float_ and np.complex_ for compatibility with libraries expecting them
np.float_ = np.float64
np.complex_ = np.complex128
