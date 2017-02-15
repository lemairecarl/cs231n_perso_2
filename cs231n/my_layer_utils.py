from layers import *

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):

  a, fc_cache = affine_forward(x, w, b)
  y, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(y)
  cache = (fc_cache, bn_cache, relu_cache)

  return out, cache


def affine_bn_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dgamma, dbeta = batchnorm_backward_alt(da, bn_cache)
  du, dw, db = affine_backward(dx, fc_cache)
  return du, dw, db, dgamma, dbeta