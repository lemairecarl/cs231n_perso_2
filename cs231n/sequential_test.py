from sequential import *
#from cs231n.data_utils import get_CIFAR10_data

# Make data
batch_shape = N, D = 20, 100
X = np.random.randn(N, D)
y = np.random.randint(10, size=N)

# Build network
model = Sequential(input_shape=batch_shape)
model.add(Dense(num_neurons=10))
model.build(loss=Softmax())

# Forward + Backward
loss, grads = model.loss(X, y)
print loss