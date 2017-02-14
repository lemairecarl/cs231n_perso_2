import numpy as np

def compute_output_size(inshape, fshape, stride):
  N, M = inshape
  HH, WW = fshape
  row_extent = (N - HH) / stride[0] + 1
  col_extent = (M - WW) / stride[1] + 1
  return row_extent, col_extent


def im2col(x, fshape, stride, verbose=False):
  # http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python

  # Parameters
  N, M = x.shape
  HH, WW = fshape
  stride_i, stride_j = stride
  row_extent, col_extent = compute_output_size(x.shape, fshape, stride)
  if verbose:
    print 'row_extent, col_extent', row_extent, col_extent

  # Get Starting block indices
  start_idx = np.arange(HH)[:, None] * M + np.arange(WW)
  if verbose:
    print start_idx

  # Get offsetted indices across the height and width of input array
  offset_idx = np.arange(row_extent)[:, None] * M * stride_i + np.arange(col_extent) * stride_j
  if verbose:
    print offset_idx

  indices = start_idx.ravel() + offset_idx.ravel()[:, None]
  if verbose:
    print indices

  # Get all actual indices & index into input array for final output
  return np.take(x, indices), indices


def sum_col2im(col, indices, inshape):
  # inshape = (H, W * C)

  # col     : (n_blocks, block_size)  block_size = HH * WW * C + 1
  # indices : (n_blocks, block_size)
  # assert col.shape == indices.shape
  result = sum_by_group(col.flatten(), indices.flatten())  # --> (input.size,) = (H * W * C)
  return np.reshape(result, inshape)


def sum_by_group(values, groups):
  # http://stackoverflow.com/questions/4373631/sum-array-by-number-in-numpy
  order = np.argsort(groups)
  groups = groups[order]
  values = values[order]
  values.cumsum(out=values)
  index = np.ones(len(groups), 'bool')
  index[:-1] = groups[1:] != groups[:-1]
  values = values[index]
  groups = groups[index]
  values[1:] = values[1:] - values[:-1]
  # assert np.array_equal(groups, np.arange(len(groups)))
  return values
