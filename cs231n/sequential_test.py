from sequential import *
from cs231n.data_utils import get_CIFAR10_data
import test


def make_data():
  # Make data
  N = 20
  X = np.random.randn(N, 3, 32, 32) * 127
  y = np.random.randint(10, size=N)
  return X, y


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
#test.overfit_small_data(model, num_train=num_train, epochs=20)

data = get_CIFAR10_data(dir='datasets/cifar-10-batches-py')
num_train = 20
num_val = np.minimum(data['X_val'].shape[0], num_train)
small_data = {
'X_train': data['X_train'][:num_train],
'y_train': data['y_train'][:num_train],
'X_val': data['X_val'][:num_val],
'y_val': data['y_val'][:num_val],
}

model = Sequential(batch_shape=small_data['X_train'].shape, weight_scale=1e-3)
model.add(Dense(num_neurons=10))
model.build(loss=Softmax())