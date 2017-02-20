from sequential import *

# Build network
model = Sequential(input_shape=(3, 32, 32))
model.add(Dense(num_neurons=10))
model.build(loss=Softmax)
# model.layers = [Dense_instance]

# Forward + Backward
X = None  # raw data
y = None  # labels
loss, grads = model.loss(X, y)
