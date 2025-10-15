"""
tests for map.py
"""

from pathlib import Path
import numpy as np
import pytest

from hw0.data import Dataset
from hw0.map import Map

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="function")
def default_empty_map() -> Map:
    DEFAULT_DIMS = np.array(
        [
            [-2, 5],
            [-6, 6],
        ]
    )
    DEFAULT_CELL_SIZE = 1.0

    cfg = Map.Config(
        DEFAULT_DIMS,
        DEFAULT_CELL_SIZE,
        np.array([0.5, -1.5]),
        np.array([0.5, 1.5]),
    )

    return Map(cfg, [])


@pytest.mark.parametrize(
    "world_coord, expected_idx", [((0, 0), (6, 2)), ((-1.5, -5.5), (0, 0))]
)
def test_world_coord_to_idx(world_coord, expected_idx, default_empty_map):
    coord = np.array(world_coord)
    loc = default_empty_map.world_coords_to_grid_index(coord)
    assert loc[0] == expected_idx[0]
    assert loc[1] == expected_idx[1]


def test_map_from_ds():
    p = REPO_ROOT / "data/ds1"
    ds = Dataset.from_dataset_directory(p)

    DEFAULT_DIMS = np.array(
        [
            [-2, 5],
            [-6, 6],
        ]
    )
    DEFAULT_CELL_SIZE = 1.0

    cfg = Map.Config(
        DEFAULT_DIMS,
        DEFAULT_CELL_SIZE,
        np.array([0.5, -1.5]),
        np.array([0.5, 1.5]),
    )

    map = Map.construct_from_dataset(ds, cfg)

    assert np.sum(map.grid == 1) == len(ds.landmarks)
