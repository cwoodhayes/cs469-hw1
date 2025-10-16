"""
tests for utils.py
"""

import numpy as np
from hw0 import utils


def test_write_square_kernel():
    # test a non-clipping example
    arr = np.zeros(shape=(8, 10))
    utils.write_square_kernel_with_clip(arr, (1, 1), 1, 1)

    assert np.all(arr[0:3, 0:3] == 1)
    assert np.all(arr[3, :] == 0) and np.all(arr[:, 3] == 0)

    # write with a clip
    arr = np.zeros(shape=(8, 10))
    utils.write_square_kernel_with_clip(arr, (0, 0), 1, 1)

    assert np.all(arr[0:2, 0:2] == 1)
    assert np.all(arr[2, :] == 0) and np.all(arr[:, 2] == 0)

    # write the whole array
    arr = np.zeros(shape=(8, 10))
    utils.write_square_kernel_with_clip(arr, (5, 6), 10, 1)

    assert np.all(arr == 1)

    # we want a radius of 0 to have the current cell
    arr = np.zeros(shape=(8, 10))
    utils.write_square_kernel_with_clip(arr, (5, 6), 0, 1)

    assert arr[5, 6] == 1
    arr[5, 6] = 0
    assert np.all(arr == 0)
