"""
one-off utility functions
"""

import typing
import numpy as np


def write_square_kernel_with_clip(
    dest: np.ndarray, center: tuple[int, int], radius: int, val: typing.Any
) -> None:
    """
    Write val in a square kernel to the destination array, ignoring any
    indices that are out of bounds of dest.
    radius=0 means a single value is written, radius=1 means a 3x3 box is written, etc

    :param dest: NxM destination to write to
    :param center: (row, col)
    :param radius: 1/2 side length
    :param val: any value of dest's dtype
    """

    xlim = (max(0, center[0] - radius), min(dest.shape[0], center[0] + radius + 1))
    ylim = (max(0, center[1] - radius), min(dest.shape[1], center[1] + radius + 1))

    dest[xlim[0] : xlim[1], ylim[0] : ylim[1]] = val
