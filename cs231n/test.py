import numpy as np
from layers import *

A = np.random.randint(0, 9, (5, 5 * 3))
print A
print np.arange(A.size).reshape((A.shape))
print im2col(A, (3, 3, 3), stride=(1, 2), verbose=True)