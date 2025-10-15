"""
Map of the world according to the robot, represented in world coordinates
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from hw0.data import Dataset


class Map:
    """
    Map of the world in a grid form
    """

    @dataclass(frozen=True)
    class Config:
        """
        :param dimensions: dimensions of the map in meters [[xmin, xmax], [ymin, ymax]]
        :type dimensions: np.ndarray[[xlim],[ylim]]
        :param cell_size: side size of an individual square cell in meters
        :type cell_size: tuple[float, float]
        :param start: start location in world coords (x,y)
        :param goal: goal location in world coords (x,y)
        """

        dimensions: np.ndarray
        cell_size: float
        start: np.ndarray
        goal: np.ndarray

    def __init__(self, config: Config, obstacles: list[np.ndarray]) -> None:
        """
        :param config: configuration for the map
        :param obstacles: Nx2 array of obstacles, where each row is an (x,y) location of an obstacle in world coords
        """
        self._c = config
        dim = np.array(config.dimensions)
        self._dim = dim
        self._world_mins = dim[:, 0]
        self._cell_size = config.cell_size
        x_range = dim[0, 1] - dim[0, 0]
        y_range = dim[1, 1] - dim[1, 0]
        range = np.array([y_range, x_range])

        shape = range / config.cell_size
        shape_int = shape.round().astype(int)
        if not np.isclose(shape_int, shape).all():
            raise ValueError(
                f"{dim} grid does not break evenly into cells of size {config.cell_size}"
            )

        self.grid = np.zeros(shape=shape_int, dtype=int)
        # maintain this list mostly for debugging
        self._obstacles = []

        self.add_obstacles(obstacles)
        start_loc = self.world_coords_to_grid_index(config.start)
        goal_loc = self.world_coords_to_grid_index(config.goal)
        self.grid[start_loc] = 2
        self.grid[goal_loc] = 3

    def world_coords_to_grid_index(self, coord: np.ndarray) -> tuple[int, int]:
        """
        Convert world coordinates (as [x,y]) into grid indices (row,col) of the cell containing that location

        :param coord: world coordinate in [x,y]
        :type coord: np.ndarray
        :return: grid indices [row, col]
        :rtype: tuple[int, int]
        """
        # todo check for invalid input
        rounds = (coord / self._cell_size - self._world_mins).round().astype(int)
        loc = (rounds[1], rounds[0])
        return loc

    def add_obstacles(self, obstacles: list[np.ndarray]) -> None:
        """
        Add a set of obstacles to the map.
        (we only ever add them, we never remove)

        :param obstacles: Nx2 list of obstacles, where each row is an (x,y) location of an obstacle in world coords
        """
        for obs in obstacles:
            row, col = self.world_coords_to_grid_index(obs)
            self.grid[row, col] = 1
            self._obstacles.append(obs)

    @classmethod
    def construct_from_dataset(cls, ds: Dataset, config: Config) -> Map:
        """
        construct a map with all the obstacles from the dataset

        :param ds: Dataset object
        :type ds: Dataset
        """
        obstacles = [
            np.array((x, y)) for x, y in zip(ds.landmarks["x_m"], ds.landmarks["y_m"])
        ]

        return cls(config, obstacles)
