"""
tests for map.py
"""
import numpy as np
import pytest

from hw0.map import Map


@pytest.fixture(scope="function")
def default_empty_map() -> Map:
    DEFAULT_DIMS=np.array([
        [-2, 5],
        [-6, 6],
    ])
    DEFAULT_CELL_SIZE = 1.0

    return Map(
        DEFAULT_DIMS,
        DEFAULT_CELL_SIZE,
        [],
        np.array([0.5, -1.5]),
        np.array([0.5, 1.5]),
    )


@pytest.mark.parametrize("world_coord, expected_idx",
                         [
                            (
                                (0, 0), (6, 2)
                            ),
                            (
                                (-1.5, -5.5), (0, 0)
                            )
                         ])
def test_world_coord_to_idx(world_coord, expected_idx, default_empty_map):
    coord = np.array(world_coord)
    loc = default_empty_map.world_coords_to_grid_index(coord)
    assert loc[0] == expected_idx[0]
    assert loc[1] == expected_idx[1]