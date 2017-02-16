import numpy as np
from cs231n.classifiers.convnet import *
from cs231n.gradient_check import eval_numerical_gradient
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


def gradient_check():
  num_inputs = 2
  input_dim = (3, 16, 16)  # 16 - 16 16 8 (32, 8, 8) - 8 8 4 (64, 4, 4) - 4 4 2 - (128, 2, 2)
  reg = 0.0
  num_classes = 10
  X = np.random.randn(num_inputs, *input_dim)
  y = np.random.randint(num_classes, size=num_inputs)
  
  model = FlexNet(input_dim=input_dim, num_filters=(4,), hidden_dim=(10,), dtype=np.float64)
  model.print_params()

  print '\n--- Gradient check ---'
  loss, grads = model.loss(X, y)
  results = {}
  smallest = {}
  for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6, pname=param_name)
    e = rel_error(param_grad_num, grads[param_name])
    results[param_name] = e
    smallest[param_name] = np.min(np.abs(np.concatenate((grads[param_name].flatten(), param_grad_num.flatten()))))

  sys.stdout.flush()
  print '\n\nMax relative error:'
  for p in sorted(results):
    print '{:<10} {:e} {:e} {}'.format(p, results[p], smallest[p], gradient_check_message(results[p]))
      

#loss_sanity_check()
gradient_check()
