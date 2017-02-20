from cs231n.solver import Solver
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


# loss_sanity_check()
# test.overfit_small_data(model, num_train=num_train, epochs=20)

total_examples = 3
data = get_CIFAR10_data(dir='datasets/cifar-10-batches-py')
X = data['X_train'][:total_examples] / 127.0
y = data['y_train'][:total_examples]

model = Sequential(batch_shape=X.shape, weight_scale=1e-3, reg=0.0)
model.add(ConvBnRelu(2))
model.add(Pool(pool_factor=8))
model.add(Dense(num_neurons=10))
model.build(loss=Softmax())
model.print_params()

print '--- Train a few epochs ---'
solver = Solver(model, {'X_train': X, 'y_train': y, 'X_val': X, 'y_val': y},
                num_epochs=20, batch_size=3,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-4,
                },
                verbose=False, print_every=1)
solver.train()
print 'Train acc', solver.train_acc_history[-1], '  Loss', solver.loss_history[-1]

test.gradient_check(model, X, y)
