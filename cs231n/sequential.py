from abc import ABCMeta, abstractmethod

import numpy as np


def is_x_a_y(x, y):
  """Returns True if x's class is a subclass of y."""
  return y in x.__class__.__bases__


class Sequential(object):
  
  num_instances = {}  # Num instances of each layer type
  
  def __init__(self, batch_shape, reg=0.0, weight_scale=1e-3):
    Sequential.num_instances.clear()
    
    self.weight_scale = weight_scale
    self.reg = reg
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
    
    # Regularization TODO really regul all params?
    for n, w in self.params.items():
      loss += 0.5 * self.reg * np.sum(np.square(w))
      self.grads[n] += self.reg * w
    
    return loss, self.grads


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
    self.model.params[name] = arr
    return self.model.params[name]
  
  def set_grad(self, name, arr):
    name = self.name + '_' + name
    self.model.grads[name] = arr
      
  
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
  def __init__(self, num_neurons):
    super(self.__class__, self).__init__()
    
    self.num_neurons = num_neurons
    self.w = None
    self.previous_output_shape = None
    self.x1 = None  # cached value

  def init(self):
    self.previous_output_shape = self.previous_layer.output_shape
    input_dim = np.prod(self.previous_output_shape[1:])
    w_b = np.random.randn(input_dim + 1, self.num_neurons) * self.model.weight_scale
    w_b[-1, :] = 0  # Init bias to zero
    self.w = self.add_param('Wb', w_b)
    
    self.output_shape = (self.previous_output_shape[0], self.num_neurons)
  
  def forward(self):
    n = self.previous_output_shape[0]
    x = self.get_input_data().reshape(n, -1)
    self.x1 = np.concatenate((x, np.ones((n, 1))), axis=1)
    self.output_data = self.x1.dot(self.w)
  
  def backward(self):
    dout = self.get_upstream_grad()
    # Grad wrt input
    dx1 = dout.dot(self.w.T)  # --> (20, 101)
    self.out_grad = dx1[:, :-1].reshape(self.previous_output_shape)
    # Grad wrt params
    self.set_grad('Wb', self.x1.T.dot(dout))


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