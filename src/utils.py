"""Some helper functions."""
import numpy as np
import logging


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


def normalise(attribute):
    """Normalize a list between 0-1."""
    _min = attribute.min()
    _min_max = attribute.max() - _min
    attribute = np.array(
        [(value - _min) / _min_max for value in attribute])
    return attribute


def remove_temp_files():
    """Remove the files generated under the HPF folder."""
    try:
        subprocess.run('rm /tmp/hpf/*.*', shell=True)
    except Exception as e:
        logger.error("Error deleting /tmp/hpf")
        logger.error("Error {}".format(e))

def convert_string2bool_env(parameter):
    """Convert the String True/False to its boolean form.

    :param parameter: The string that needs to be converted.
    """
    return parameter.lower() == "true"
