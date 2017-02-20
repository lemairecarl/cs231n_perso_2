from abc import ABCMeta, abstractmethod


def is_x_a_y(x, y):
  """Returns True if x's class is a subclass of y."""
  return y in x.__class__.__bases__


class Sequential(object):
  def __init__(self, input_shape):
    self.layers = []
    self.input_shape = input_shape
    self.loss_layer = None
    self.params = None
    self.grads = None
    
    self.input_layer = InputLayer()
    self.add(self.input_layer)  # TODO input shape?
    
  def add(self, layer):
    if not is_x_a_y(layer, SequentialLayer):
      raise TypeError("parameter must be a SequentialLayer")
    
    layer.model = self
    self.layers.append(layer)
    
  def build(self, loss):
    self.loss_layer = loss
    # TODO init params
    
  def loss(self, X, y=None):
    # Forward pass
    self.input_layer.feed(X)
    for l in self.layers:
      l.forward()
    
    if y is not None:
      # Backward pass
      self.loss_layer.set_ground_truth(y)
      for l in reversed(self.layers):
        l.backward()
  
      # TODO regularization
    
    return self.loss_layer.loss, self.grads


class SequentialLayer:
  __metaclass__ = ABCMeta
  
  def __init__(self):
    self.model = None
    self.output = None
    self.gradient = None
    self.output_data = None
    
  @abstractmethod
  def forward(self):
    """
    Compute output and backward pass cache, given data (and params which are already known).
    Gets input from previous layer.
    Puts output in the output attribute.
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
  
  
class InputLayer(SequentialLayer):
  def forward(self):
    pass

  def backward(self):
    pass
  
  def feed(self, data):
    self.output_data = data
