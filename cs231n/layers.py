import numpy as np
from blocks import *

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = x.shape[0]
  D = np.prod(x.shape[1:])
  X = np.reshape(x, (N, D))

  # Compute output
  out = X.dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


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
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  N = x.shape[0]
  D = np.prod(x.shape[1:])
  X = np.reshape(x, (N, D))

  dx = dout.dot(w.T).reshape(x.shape)
  dw = X.T.dot(dout)
  db = np.sum(dout, axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x > 0
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x_gt_zero = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dout_dx = x_gt_zero
  dx = dout * dout_dx
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    # Compute mean and var
    batch_mean = np.mean(x, axis=0)  # shape (D,)
    batch_var = np.var(x, axis=0)  # shape (D,)
    # Compute output
    delta = x - batch_mean
    inv_var = 1.0 / (batch_var + eps)
    inv_sqrt_var = np.sqrt(inv_var)
    bn = delta * inv_sqrt_var  # batch norm
    out = gamma[np.newaxis, :] * bn + beta[np.newaxis, :]  # scaling
    cache = {
      'batch_mean': batch_mean,
      'batch_var': batch_var,
      'delta': delta,
      'inv_var': inv_var,
      'inv_sqrt_var': inv_sqrt_var,
      'bn': bn,
      'gamma': gamma
    }
    # Update running mean and var
    running_mean = momentum * running_mean + (1 - momentum) * batch_mean
    running_var = momentum * running_var + (1 - momentum) * batch_var
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    out = (x - running_mean) / np.sqrt(running_var + eps)  # batch norm
    out = gamma * out + beta  # scaling
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  N, D = dout.shape
  gamma = cache['gamma']
  inv_sqrt_var = cache['inv_sqrt_var']

  dout_dxn = dout * gamma  # (N, D)
  dout_dv = np.sum(dout_dxn * cache['delta'], axis=0) * (-1.0 / 2) * inv_sqrt_var ** 3  # (D,)
  dout_ddelta_v = dout_dv * (2.0 / N) * cache['delta']  # (N, D)
  dout_ddelta_d = dout_dxn * inv_sqrt_var  # (N, D)
  dout_dmu_d = -1 * np.sum(dout_ddelta_d, axis=0)  # (D,)
  dout_dmu_v = -1 * np.sum(dout_ddelta_v, axis=0)  # (D,)
  dout_dx_m = (dout_dmu_d + dout_dmu_v) * (1.0 / N)  # (D,)
  dout_dx_0 = dout_ddelta_d + dout_ddelta_v  # (N, D)
  dx = dout_dx_0 + dout_dx_m

  dout_dgamma = cache['bn']
  dgamma = np.sum(dout * dout_dgamma, axis=0)

  dout_dbeta = np.array(1)
  dbeta = np.sum(dout * dout_dbeta, axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  N, D = dout.shape

  gamma = cache['gamma']
  inv_var = cache['inv_var']
  inv_sqrt_var = cache['inv_sqrt_var']
  delta = cache['delta']
  term1 = N * dout
  term2 = np.sum(dout, axis=0)
  term3 = delta * inv_var * np.sum(dout * delta, axis=0)
  dx = (1.0 / N) * gamma * inv_sqrt_var * (term1 - term2 - term3)

  dout_dgamma = cache['bn']
  dgamma = np.sum(dout * dout_dgamma, axis=0)

  dout_dbeta = np.array(1)
  dbeta = np.sum(dout * dout_dbeta, axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p_drop, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    p = 1.0 - p_drop
    mask = (np.random.rand(*x.shape) < p) / p
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']

  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param, verbose=0):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  if verbose > 0:
    print 'Before pad', x.shape
  p = conv_param['pad']
  x = np.pad(x, [(0, 0), (0, 0), (p, p), (p, p)], mode='constant')  # pad with zeros
  if verbose > 0:
    print 'After pad', x.shape
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  stride = (conv_param['stride'],) * 2

  # Flatten filters as columns in a matrix
  w_col = np.reshape(w, (F, -1))  # --> (F, fsize) where fsize = C * HH * WW
  w_col = w_col.T  # make compatible for matrix mult --> (fsize, F)
  w_col = np.concatenate((w_col, b[None, :]), axis=0)  # include weights! --> (fsize + 1, F)
  if verbose > 0:
    print 'w_col', w_col.shape
  row_extent, col_extent = compute_output_size(x.shape[2:], (HH, WW), stride)
  num_blocks = row_extent * col_extent
  if verbose > 0:
    print 'row_extent, col_extent', row_extent, col_extent

  blocks_with_bias = np.empty((N, num_blocks, w_col.shape[0]))
  im2col_indices = np.empty((N, num_blocks, w_col.shape[0] - 1))  # Bias not in this
  a_col = np.empty((N, num_blocks, F))
  if verbose > 0:
    print 'a_col', a_col.shape
  for i, image in enumerate(x):
    im_col, im2col_indices[i, :, :] = im3d_to_col(image, (C, HH, WW), stride=stride)  # make blocks, keep indices for backpr
    im_col = np.concatenate((im_col, np.ones((num_blocks, 1))), axis=1)  # include bias factor
    blocks_with_bias[i, :, :] = im_col  # (n_blocks, fsize + 1 + 1)
    if verbose > 1:
      print 'im_col', im_col.shape
    a_col[i, :, :] = im_col.dot(w_col)

  # Reshape activations from 1D to 3D
  # a_col : (N, n_blocks, F)
  a = np.moveaxis(a_col, -1, 1)  # --> (N, F, n_blocks)
  if verbose > 0:
    print a.shape
  out = np.reshape(a, (N, F, row_extent, col_extent))
  if verbose > 0:
    print out.shape
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (
    blocks_with_bias,
    w_col,  # flattened filters with bias
    im2col_indices,
    x.shape,  # padded
    w.shape,
    conv_param
  )
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.  (N, F, H', W')
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x : (N, C, H, W)
  - dw: Gradient with respect to w : (F, C, HH, WW)
  - db: Gradient with respect to b : (F,)
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################

  blocks_with_bias, w_col, im2col_indices, x_shape, w_shape, conv_param = cache
  # blocks_with_bias (X) has shape (N, n_blocks, HH * WW * C + 1)

  N, C, H, W = x_shape  # padded
  F, C, HH, WW = w_shape

  # For each image i in X:
  #   dx = dA dot W.T
  #   dW = x.T dot dA
  #   where   x is the blocks of image i with biases  (n_blocks, fsize + 1)
  #           W is the weights with biases            (fsize + 1, n_filters)
  #           A is the activations (out)              (n_blocks, n_filters)

  n_blocks = blocks_with_bias.shape[1]
  dout = np.reshape(dout, (N, F, n_blocks))
  dout = np.moveaxis(dout, 1, -1)  # --> (N, n_blocks, F)

  pad = conv_param['pad']
  dx = np.zeros((N, C, H - 2 * pad, W - 2 * pad))
  dw = np.zeros(w_shape)
  db = np.zeros(w_shape[0])

  for i, x in enumerate(blocks_with_bias):
    # x : (n_blocks, C * HH * WW + 1)

    # compute gradient wrt weights and biases
    image_dW = x.T.dot(dout[i])

    # extract dw and db
    dw_flat = image_dW[:-1, :]  # --> (C * HH * WW, F)
    dw_flat = dw_flat.T  # --> (F, C * HH * WW)
    image_dw = np.reshape(dw_flat, (F, C, HH, WW))
    dw += image_dw
    db += image_dW[-1, :]

    # compute block-wise gradient : (n_blocks, C * HH * WW + 1) per image
    image_dX = dout[i].dot(w_col.T)

    # Discard gradient wrt 1-column
    image_dX = image_dX[:, :-1]  # --> (n_blocks, C * HH * WW)

    # Get gradients wrt pixel components
    dpix = sum_by_group(image_dX.flatten(), im2col_indices[i].flatten())  # --> (C * H * W)
    image_dx = np.reshape(dpix, (C, H, W))
    image_dx = image_dx[:, pad:-pad, pad:-pad]  # unpad
    dx[i, :, :, :] = image_dx

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  pool_size = pool_param['pool_height'], pool_param['pool_width']
  stride = (pool_param['stride'],) * 2
  out_size = x.shape / np.array((1, 1) + pool_size)  # = (N, C, H', W')
  n_blocks = np.prod(out_size[-2:])
  block_size = int(np.prod(pool_size))

  out = np.empty(out_size)
  orig_idx = np.empty((N, np.prod(out_size[1:])), dtype=np.uint32)
  for i, activation in enumerate(x):
    # activation : (C, H, W)
    # Convert input to block columns
    x_col, im2col_indices = im3d_to_col(activation, (1,) + pool_size, stride)  # --> (C * n_blocks, block_size)
    col_max_idx = np.argmax(x_col, axis=1)
    max_mask = np.arange(block_size)[None, :] == col_max_idx[:, None]
    out_flat = x_col[max_mask]  # (C * H' * W')
    orig_idx[i, :] = im2col_indices[max_mask]  # (C * H' * W')
    out_3d = np.reshape(out_flat, out_size[1:])
    out[i] = out_3d
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x.shape, orig_idx)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x_shape, orig_idx = cache

  block3d_size = np.prod(x_shape[1:])
  offsets = np.arange(x_shape[0], dtype=np.uint32) * block3d_size
  orig_idx += offsets[:, None]  # idx rel to img --> idx rel to mini-batch

  dx = np.zeros(x_shape)
  np.put(dx, orig_idx, dout.flatten())
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  # Reshape input data for the shape accepted by batchnorm (N', D)
  # N' = N * H * W
  # D  = C
  N, C, H, W = x.shape
  x_2d = np.moveaxis(x, 1, -1)
  x_2d = np.reshape(x_2d, (N * H * W, C))
  out_2d, cache = batchnorm_forward(x_2d, gamma, beta, bn_param)
  out = np.reshape(out_2d, (N, H, W, C))
  out = np.moveaxis(out, -1, 1)  # --> (N, C, H, W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = dout.shape
  dout_2d = np.moveaxis(dout, 1, -1)
  dout_2d = np.reshape(dout_2d, (N * H * W, C))
  dx_2d, dgamma, dbeta = batchnorm_backward_alt(dout_2d, cache)
  dx = np.reshape(dx_2d, (N, H, W, C))
  dx = np.moveaxis(dx, -1, 1)  # --> (N, C, H, W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y, scale=1.0):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  
  loss *= scale
  dx *= scale  # useful for gradient checking
  return loss, dx