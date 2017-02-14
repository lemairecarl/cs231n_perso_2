import numpy as np
from layers import *
from blocks import *

# A = np.random.randint(0, 9, (5, 5 * 3))
# print A
# print np.arange(A.size).reshape((A.shape))
# print im2col(A, (3, 3, 3), stride=(1, 2), verbose=True)

# x_shape = (2, 3, 4, 4)
# w_shape = (3, 3, 4, 4)
# x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
# w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
# b = np.linspace(-0.1, 0.2, num=3)
#
# conv_param = {'stride': 2, 'pad': 1}
# out, _ = conv_forward_naive(x, w, b, conv_param, verbose=1)

a = np.random.randint(0, 9, (4, 6))
print a
block_view = im2col(a, (2, 4), (1, 2))
