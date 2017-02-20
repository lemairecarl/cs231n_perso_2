from sequential import *
#from cs231n.data_utils import get_CIFAR10_data
import test


def make_data():
  # Make data
  N = 20
  X = np.random.randn(N, 3, 32, 32) * 127
  y = np.random.randint(10, size=N)
  return X, y

# Build network
num_train = 10
model = Sequential(batch_shape=(num_train, 3, 32, 32))
model.add(Dense(num_neurons=10))
model.build(loss=Softmax())


def loss_sanity_check():
  X, y = make_data()
  # Build network
  model = Sequential(batch_shape=X.shape)
  model.add(Dense(num_neurons=10))
  model.build(loss=Softmax())
  # Forward + Backward
  loss, grads = model.loss(X, y)
  print '--- Loss sanity check ---'
  print loss


#loss_sanity_check()
test.overfit_small_data(model, num_train=num_train, epochs=20)
