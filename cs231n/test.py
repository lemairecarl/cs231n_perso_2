import numpy as np
from cs231n.classifiers.convnet import *
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient2
from cs231n.solver import Solver
from cs231n.data_utils import get_CIFAR10_data
import sys

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def loss_sanity_check():
  print '\n--- Loss sanity check ---'
  
  model = FlexNet(num_filters=(32, 64, 128), hidden_dim=(100, 50))
  
  N = 50
  X = np.random.randn(N, 3, 32, 32)
  y = np.random.randint(10, size=N)
  loss, grads = model.loss(X, y)
  print 'Initial loss (no regularization): ', loss
  
  model.reg = 0.5
  loss, grads = model.loss(X, y)
  print 'Initial loss (with regularization): ', loss


def gradient_check_message(err):
  if err > 1e-2:
    return 'probably wrong'
  elif err > 1e-4:
    return 'uncomfortable'
  elif err < 1e-7:
    return 'great'
  else:
    return ''


def fix_kinks(grad_ana, grad_num):
  nan_mask = grad_num > 1e90
  num_kinks = np.sum(nan_mask)
  if num_kinks > 0:
    print num_kinks, 'kinks encountered over', grad_ana.size
    grad_num[nan_mask] = grad_ana[nan_mask]  # Where kinks are crossed, treat as error = 0
  return num_kinks


def gradient_check(model, X, y):
  # num_inputs = 2
  # input_dim = (3, 32, 32)
  # reg = 0.0
  # num_classes = 10
  # X = np.random.randn(num_inputs, *input_dim)
  # y = np.random.randint(num_classes, size=num_inputs)
  #
  # model = FlexNet(input_dim=input_dim, num_filters=(4,), hidden_dim=(10,), reg=reg, dtype=np.float64)
  # model.print_params()
  #
  # # Train a bit before grad check
  # model = overfit_small_data(model, epochs=4, verbose=False)
  #
  #model.loss_scale = 1e4
  #model.compute_hashes = True
  
  # TODO functional model
  # TODO check individual parts?
  # TODO check fewer dimensions
  # TODO test without reg and only reg
  # TODO try multiple h
  
  print '\n--- Gradient check ---'
  loss, grads = model.loss(X, y)
  results = {}
  avg = {}
  h = 1e-6
  for param_name in sorted(grads):
    def f(_):
      out = model.loss(X, y)
      return out[0]
    
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=h)
    avg[param_name] = np.mean(np.abs(grads[param_name])), np.mean(np.abs(param_grad_num))
    results[param_name] = rel_error(param_grad_num, grads[param_name])
  
  sys.stdout.flush()
  print 'Max relative error:   (h = {})'.format(h)
  print '{:<20} {:<13} {:<15}           {:<13} {:<13}'.format('Param', 'Error', '', 'Ana', 'Num')
  for p in sorted(results):
    msg = gradient_check_message(results[p])
    print '{:<20} {:<13e} {:<15}   avgval: {:<13e} {:<13e}'.format(p, results[p], msg, avg[p][0], avg[p][1])


def overfit_small_data(model=None, epochs=10, num_train=20, verbose=True):
  
  data = get_CIFAR10_data(dir='datasets/cifar-10-batches-py')
  small_data = {
    'X_train': data['X_train'][:num_train] / 127.0,
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'][:num_train] / 127.0,  # batch size must be constant
    'y_val': data['y_val'][:num_train],
  }

  if model is None:
    input_dim = small_data['X_train'].shape[1:]
    print input_dim
    # 32 - 16, 8, 4, 2
    model = FlexNet(input_dim=input_dim, num_filters=(8, 8, 16, 16), hidden_dim=(100,))
    model.print_params()

  print '\n--- Training a few epochs ---'
  
  solver = Solver(model, small_data,
                  num_epochs=epochs, batch_size=np.minimum(50, num_train),
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 1e-4,
                  },
                  verbose=verbose, print_every=1)
  solver.train()
  print 'Train acc:', solver.train_acc_history[-1]
  return model
   

if __name__ == "__main__":
  #loss_sanity_check()
  #gradient_check()
  overfit_small_data()
