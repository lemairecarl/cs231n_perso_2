import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from cs231n.my_layer_utils import *


def pn(p, i):
  return p + str(i)


class FlexNet(object):
  """
  [conv-relu-conv-relu-pool]xN - [affine-relu]xM - affine - [softmax or SVM]

  conv are all of type 'same'

  N = len(num_filters) -- num_filter is the same for both inside a block
  M = len(hidden_dim)
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=(32,), filter_size=3,
               hidden_dim=(100,), num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.layers = {}
    
    # pass conv_param to the forward pass for the convolutional layer
    self.conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    
    # pass pool_param to the forward pass for the max-pooling layer
    self.poolsize = 2
    
    # Parameter init ===================
    
    # Parameter names: (W | b) + (co | af) [+ (a | b)] + j
    #     where   i : block kind : conv-conv-pool or affine
    #             j : 0-based index of the block instance
    
    C, H, W = input_dim
    
    self.params = {}
    # Params for first repeatable block (conv-conv-pool)
    image_size = np.array((H, W))
    self.layers['phase1'] = []
    for i, block_num_filters in enumerate(num_filters):
      self.layers['phase1'] += [None]
      self.params[pn('Wcoa', i)] = np.random.randn(block_num_filters, C, filter_size, filter_size) * weight_scale
      self.params[pn('Wcob', i)] = np.random.randn(block_num_filters, C, filter_size, filter_size) * weight_scale
      self.params[pn('bcoa', i)] = np.zeros(block_num_filters)
      self.params[pn('bcob', i)] = np.zeros(block_num_filters)
      image_size /= self.poolsize  # Effect of pooling layer
    
    # Params for last repeatable block (affine)
    input_size = (num_filters[-1] * np.prod(image_size),) + hidden_dim
    self.layers['phase2'] = []
    for i, block_hidden_dim in enumerate(hidden_dim):
      self.layers['phase2'] += [None]
      self.params[pn('Waf', i)] = np.random.randn(input_size[i], block_hidden_dim) * weight_scale
      self.params[pn('baf', i)] = np.zeros(block_hidden_dim)
      self.params[pn('gamma_af', i)] = np.ones((block_hidden_dim,))
      self.params[pn('beta_af', i)] = np.zeros((block_hidden_dim,))
    
    # Params for last affine layer
    self.params['Wlast'] = np.random.randn(input_size[-1], num_classes) * weight_scale
    self.params['blast'] = np.zeros(num_classes)
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
  
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the convolutional network.

    Inputs:
    - X: Array of input data of shape (N, C, H, W)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
    """
    
    N, C, H, W = X.shape
    
    convp = self.conv_param
    poolp = {'pool_height': self.poolsize, 'pool_width': self.poolsize, 'stride': self.poolsize}
    bnp = {'mode': 'train' if y is not None else 'test'}
    
    F_phase1 = {'a': {}, 'b': {}, 'p': {-1: X}}  # p is pool
    c_phase1 = {'a': {}, 'b': {}, 'p': {}}
    for i, l in enumerate(self.layers['phase1']):
      # TODO implement conv_bn_relu
      F_phase1['a'][i], c_phase1['a'][i] = conv_relu_forward(F_phase1['p'][i - 1], self.getp('Wcoa', i),
                                                             self.getp('bcoa', i), convp)
      F_phase1['b'][i], c_phase1['b'][i] = conv_relu_forward(F_phase1['a'][i], self.getp('Wcob', i),
                                                             self.getp('bcob', i), convp)
      F_phase1['p'][i], c_phase1['p'][i] = max_pool_forward_fast(F_phase1['b'][i], poolp)
    
    F_phase2 = {-1: F_phase1['p'][-1]}
    c_phase2 = {'a': {}, 'b': {}, 'p': {}}
    for i, l in enumerate(self.layers['phase2']):
      F_phase2[i], c_phase2[i] = affine_bn_relu_forward(F_phase2[i - 1], self.getp('Waf', i), self.getp('baf', i),
                                                        self.getp('gamma_af', i), self.getp('beta_af', i), bnp)
    
    scores, cache_last = affine_forward(F_phase2[-1], self.params['Wlast'], self.params['blast'])
    
    if y is None:
      return scores
    
    loss, grads = 0, {}

    loss, dscores = softmax_loss(scores, y)
    dFlast_input, grads['Wlast'], grads['blast'] = affine_backward(dscores, cache_last)
    
    dF_phase2 = {len(self.layers['phase2']): dFlast_input}
    for i in range(len(self.layers['phase2']) - 1, -1, -1):
      out = affine_bn_relu_backward(dF_phase2[i + 1], c_phase2[i])
      (
        dF_phase2[i],
        grads[pn('Waf', i)],
        grads[pn('baf', i)],
        grads[pn('gamma_af', i)],
        grads[pn('beta_af', i)]
      ) = out

      # Regularization
      loss += 0.5 * self.reg * np.sum(np.square(self.getp('Waf', i)))
      grads[pn('Waf', i)] += self.reg * np.sum(self.getp('Waf', i))
      
    dF_phase1 = {'a': {len(self.layers['phase1']): dF_phase2[0]}, 'b': {}, 'p': {}}
    for i in range(len(self.layers['phase1']) - 1, -1, -1):
      dF_phase1['p'][i] = max_pool_backward_fast(dF_phase1['a'][i + 1], c_phase1['p'][i])
      dF_phase1['b'][i], grads[pn('Wcob', i)], grads[pn('bcob', i)] = conv_relu_pool_backward(dF_phase1['p'][i], c_phase1['b'][i])
      dF_phase1['a'][i], grads[pn('Wcoa', i)], grads[pn('bcoa', i)] = conv_relu_pool_backward(dF_phase1['b'][i], c_phase1['a'][i])

      # Regularization
      loss += 0.5 * self.reg * (np.sum(np.square(self.getp('Wcoa', i))) + np.sum(np.square(self.getp('Wcob', i))))
      grads[pn('Wcoa', i)] += self.reg * np.sum(self.getp('Wcoa', i))
      grads[pn('Wcob', i)] += self.reg * np.sum(self.getp('Wcob', i))
    
    return loss, grads
  
  def getp(self, p, i):
    return self.params[pn(p, i)]
