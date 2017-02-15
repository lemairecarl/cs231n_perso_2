import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def pn(p, i):
  return p + str(i)


class FlexNet(object):
  """
  [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]

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
    for i, block_num_filters in enumerate(num_filters):
      self.params[pn('Wcoa', i)] = np.random.randn(block_num_filters, C, filter_size, filter_size) * weight_scale
      self.params[pn('Wcob', i)] = np.random.randn(block_num_filters, C, filter_size, filter_size) * weight_scale
      self.params[pn('bcoa', i)] = np.zeros(block_num_filters)
      self.params[pn('bcob', i)] = np.zeros(block_num_filters)
      image_size /= self.poolsize  # Effect of pooling layer

    # Params for last repeatable block (affine)
    input_size = (num_filters[-1] * np.prod(image_size),) + hidden_dim
    for i, block_hidden_dim in enumerate(hidden_dim):
      self.params[pn('Waf', i)] = np.random.randn(input_size[i], block_hidden_dim) * weight_scale
      self.params[pn('baf', i)] = np.zeros(block_hidden_dim)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Inputs:
    - X: Array of input data of shape (N, C, H, W)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    N, C, H, W = X.shape

    F1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    F2, cache2 = affine_relu_forward(F1, W2, b2)
    scores, cache3 = affine_forward(F2, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    dF2, grads['W3'], grads['b3'] = affine_backward(dscores, cache3)
    dF1, grads['W2'], grads['b2'] = affine_relu_backward(dF2, cache2)
    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dF1, cache1)

    # Regularization
    loss += 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    grads['W1'] += self.reg * np.sum(W1)
    grads['W2'] += self.reg * np.sum(W2)
    grads['W3'] += self.reg * np.sum(W3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads