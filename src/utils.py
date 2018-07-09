"""Some helper functions."""

import numpy as np


def cal_sparsity(array):
    """Return the sparsity of the array."""
    sparsity = 1
    try:
        num_total = total_elems(array)
        sparsity = zero_elems(array) / num_total
    except ZeroDivisionError:
        pass
    return sparsity


def zero_elems(array):
    """Return a count of zero elements in an array."""
    return np.count_nonzero(array == 0)


def total_elems(array):
    """Return the total number of elements in an array."""
    shape = array.shape
    return shape[0] * shape[1]


def non_zero_entries(mat):
    """Return the row-cloumn index tuple of non zero array elements."""
    # Takes a 2 dimensional numpy array.

    indices = []
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if mat[i, j] > 0:
                indices.append((i, j))
    return tuple(indices)
