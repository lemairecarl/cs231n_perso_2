from abc import ABCMeta, abstractmethod

import numpy as np

from cs231n.fast_layers import *
from cs231n.my_layer_utils import conv_bn_relu_forward, conv_bn_relu_backward


def is_x_a_y(x, y):
  """Returns True if x's class is a subclass of y."""
  return y in x.__class__.__bases__


class Sequential(object):
  
  num_instances = {}  # Num instances of each layer type
  
  def __init__(self, batch_shape, reg=0.0, weight_scale=1e-3, dtype=np.float32):
    Sequential.num_instances.clear()
    
    self.weight_scale = weight_scale
    self.reg = reg
    self.dtype = dtype
    self.params = {}
    self.grads = {}
    
    self.layers = []
    self.loss_layer = None
    self.input_layer = InputLayer(batch_shape)
    self.add(self.input_layer)
    
  def add(self, layer):
    # TODO not working with iPython
    #if not is_x_a_y(layer, SequentialLayer):
    #  raise TypeError("parameter must be a SequentialLayer")
    
    layer.model = self
    if len(self.layers) > 0:
      layer.previous_layer = self.layers[-1]
      self.layers[-1].next_layer = layer
    self.layers.append(layer)
    
  def build(self, loss):
    self.loss_layer = loss
    self.add(self.loss_layer)
    
    for l in self.layers:
      l.init()
    
  def loss(self, X, y=None):
    # Forward pass
    assert X.shape == self.input_layer.output_shape
    self.input_layer.output_data = X
    for l in self.layers:
      l.forward()
    
    if y is None:
      return self.loss_layer.scores
      
    # Backward pass
    self.loss_layer.ground_truth = y
    for l in reversed(self.layers):
      l.backward()
    loss = self.loss_layer.loss
    
    # Regularization
    for n, w in self.params.items():
      if n.endswith('_W'):
        loss += 0.5 * self.reg * np.sum(np.square(w))
        self.grads[n] += self.reg * w
      elif n.endswith('_Wb'):
        loss += 0.5 * self.reg * np.sum(np.square(w[:-1]))  # Omit the bias
        self.grads[n][:-1] += self.reg * w[:-1]
    
    return loss, self.grads
  
  def print_params(self):
    print 'Model parameters:'
    num_params = 0
    for n in sorted(self.params):
      print '{:<20} {}'.format(n, self.params[n].shape)
      num_params += self.params[n].size
    print 'Total', num_params
    
  def print_layers(self):
    print 'Layer outputs:'
    for l in self.layers:
      print '{:<15}  out: {}'.format(l.name, l.output_shape)


class SequentialLayer:
  __metaclass__ = ABCMeta
  
  def __init__(self):
    self.model = None
    self.previous_layer = None
    self.next_layer = None
    self.name = self.make_name()
    
    self.out_grad = None
    self.output_data = None
    self.output_shape = None
    self.cache = None
    
  def make_name(self):
    counts = Sequential.num_instances
    if self.__class__ not in counts:
      counts[self.__class__] = 0
    counts[self.__class__] += 1
    num = counts[self.__class__]
    return self.__class__.__name__ + str(num)
    
  @abstractmethod
  def init(self):
    """Initialize parameters"""
    pass
    
  @abstractmethod
  def forward(self):
    """
    Compute output and backward pass cache, given data (and params which are already known).
    Gets input from previous layer.
    Puts output in the output_data attribute.
    """
    pass
  
  @abstractmethod
  def backward(self):
    """
    Compute gradients wrt inputs, given upstream gradient.
    Gets upstream gradient from next layer.
    Puts gradient in the gradient attribute.
    """
    pass
  
  def get_input_data(self):
    return self.previous_layer.output_data
  
  def get_upstream_grad(self):
    return self.next_layer.out_grad

  def add_param(self, name, arr):
    name = self.name + '_' + name
    if name in self.model.params:
      raise KeyError('Param already exists')
    self.model.params[name] = arr.astype(self.model.dtype)
    return self.model.params[name]
  
  def add_params(self, param_dict):
    for k, v in param_dict.items():
      self.add_param(k, v)
  
  def get_param(self, name):
    name = self.name + '_' + name
    return self.model.params[name]
  
  def get_params(self, names):
    params = []
    for n in names:
      name = self.name + '_' + n
      params.append(self.model.params[name])
    return tuple(params)
  
  def set_grad(self, name, arr):
    name = self.name + '_' + name
    self.model.grads[name] = arr
    
  def set_grads(self, grad_dict):
      for k, v in grad_dict.items():
        self.set_grad(k, v)
  
class InputLayer(SequentialLayer):
  def __init__(self, output_shape):
    super(self.__class__, self).__init__()
    
    self.output_shape = output_shape

  def init(self):
    pass
  
  def forward(self):
    pass

  def backward(self):
    pass


class Dense(SequentialLayer):
  def __init__(self, num_neurons, weight_init='inv_sqrt_n'):
    super(self.__class__, self).__init__()
    
    self.num_neurons = num_neurons
    self.previous_output_shape = None
    self.weight_init = weight_init

  def init(self):
    self.previous_output_shape = self.previous_layer.output_shape
    input_dim = np.prod(self.previous_output_shape[1:])
    w = np.random.randn(input_dim, self.num_neurons) * self.model.weight_scale
    if self.weight_init == 'inv_sqrt_n':
      w /= np.sqrt(input_dim)
    elif self.weight_init == 'sqrt_2_over_n':
      w *= np.sqrt(2.0 / input_dim)
    else:
      raise ValueError('WTF is that argument')
    b = np.zeros(self.num_neurons)
    self.add_params({'W': w, 'b': b})
    
    self.output_shape = (self.previous_output_shape[0], self.num_neurons)
  
  def forward(self):
    def affine_forward(x, w, b):
      """
      Computes the forward pass for an affine (fully-connected) layer.

      The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
      We multiply this against a weight matrix of shape (D, M) where
      D = \prod_i d_i

      Inputs:
      x - Input data, of shape (N, d_1, ..., d_k)
      w - Weights, of shape (D, M)
      b - Biases, of shape (M,)

      Returns a tuple of:
      - out: output, of shape (N, M)
      - cache: (x, w, b)
      """
      out = x.reshape(x.shape[0], -1).dot(w) + b
      cache = (x, w, b)
      return out, cache
    
    x = self.get_input_data()
    w, b = self.get_params(['W', 'b'])
    self.output_data, self.cache = affine_forward(x, w, b)
  
  def backward(self):
    def affine_backward(dout, cache):
      """
      Computes the backward pass for an affine layer.

      Inputs:
      - dout: Upstream derivative, of shape (N, M)
      - cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)

      Returns a tuple of:
      - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
      - dw: Gradient with respect to w, of shape (D, M)
      - db: Gradient with respect to b, of shape (M,)
      """
      x, w, b = cache
      dx = dout.dot(w.T).reshape(x.shape)
      dw = x.reshape(x.shape[0], -1).T.dot(dout)
      db = np.sum(dout, axis=0)
      return dx, dw, db
    
    dout = self.get_upstream_grad()
    self.out_grad, dw, db = affine_backward(dout, self.cache)
    self.set_grads({'W': dw, 'b': db})


class ConvBnRelu(SequentialLayer):
  def __init__(self, num_filters, filter_size=3):
    super(ConvBnRelu, self).__init__()

    self.num_filters = num_filters
    self.filter_size = filter_size
    self.cache = None
    self.conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}  # convtype: same
    self.bn_param = {'mode': 'train'}

  def init(self):
    in_shape = self.previous_layer.output_shape
    w_filters = np.random.randn(self.num_filters, in_shape[1],
                                  self.filter_size, self.filter_size) * self.model.weight_scale
    w_filters *= np.sqrt(2.0 / np.prod(in_shape[1:]))
    self.add_param('W', w_filters)
    self.add_param('b', np.zeros(self.num_filters))
    self.add_param('gamma', np.ones(self.num_filters))
    self.add_param('beta', np.zeros(self.num_filters))
    
    self.output_shape = (in_shape[0], self.num_filters, in_shape[2], in_shape[3])

  def forward(self):
    # TODO mode=test if called with y=None
    x = self.get_input_data()
    w, b, gamma, beta = self.get_params(['W', 'b', 'gamma', 'beta'])
    self.output_data, self.cache = conv_bn_relu_forward(x, w, b, gamma, beta, self.conv_param, self.bn_param)

  def backward(self):
    dout = self.get_upstream_grad()
    out = conv_bn_relu_backward(dout, self.cache)
    dx, dw, db, dgamma, dbeta = out
    self.out_grad = dx
    self.set_grads({
      'W': dw,
      'b': db,
      'gamma': dgamma,
      'beta': dbeta
    })


class Pool(SequentialLayer):
  def __init__(self, pool_factor=2):
    super(Pool, self).__init__()

    self.pool_param = {
      'pool_height': pool_factor,
      'pool_width': pool_factor,
      'stride': pool_factor
    }

  def init(self):
    ph, pw = self.pool_param['pool_height'], self.pool_param['pool_width']
    self.output_shape = tuple(self.previous_layer.output_shape / np.array((1, 1, ph, pw)).flatten())

  def forward(self):
    x = self.get_input_data()
    self.output_data, self.cache = max_pool_forward_fast(x, self.pool_param)

  def backward(self):
    dout = self.get_upstream_grad()
    self.out_grad = max_pool_backward_fast(dout, self.cache)


class Softmax(SequentialLayer):
  def __init__(self):
    super(self.__class__, self).__init__()
    
    self.loss = None
    self.scores = None
    self.ground_truth = None
    
  def init(self):
    pass
  
  def forward(self):
    self.scores = self.get_input_data()

  def backward(self):
    x = self.get_input_data()
    y = self.ground_truth
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    n = x.shape[0]
    self.loss = -np.sum(np.log(probs[np.arange(n), y])) / n
    dx = probs.copy()
    dx[np.arange(n), y] -= 1
    dx /= n
    self.out_grad = dx
    

class Relu(SequentialLayer):
  def init(self):
    self.output_shape = self.previous_layer.output_shape

  def forward(self):
    def relu_forward(x):
      """
      Computes the forward pass for a layer of rectified linear units (ReLUs).

      Input:
      - x: Inputs, of any shape

      Returns a tuple of:
      - out: Output, of the same shape as x
      - cache: x
      """
      out = np.maximum(0, x)
      cache = x
      return out, cache
    
    self.output_data, self.cache = relu_forward(self.get_input_data())

  def backward(self):
    def relu_backward(dout, cache):
      """
      Computes the backward pass for a layer of rectified linear units (ReLUs).

      Input:
      - dout: Upstream derivatives, of any shape
      - cache: Input x, of same shape as dout

      Returns:
      - dx: Gradient with respect to x
      """
      x = cache
      dx = np.where(x > 0, dout, 0)
      return dx
    
    self.out_grad = relu_backward(self.get_upstream_grad(), self.cache)