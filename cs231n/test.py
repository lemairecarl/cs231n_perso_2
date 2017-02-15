import numpy as np
from cs231n.classifiers.convnet import *

model = FlexNet(num_filters=(32, 64, 128), hidden_dim=(100, 50))

# Sanity check loss
N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)
loss, grads = model.loss(X, y)
print loss