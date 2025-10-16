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
        self.c = config
        dim = self.c.dimensions
        self._world_mins = dim[:, 0]
        x_range = dim[0, 1] - dim[0, 0]
        y_range = dim[1, 1] - dim[1, 0]
        # flip x, y to be row, col
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
        self._start_loc = self.world_coords_to_grid_index(config.start)
        self._goal_loc = self.world_coords_to_grid_index(config.goal)
        self.grid[self._start_loc] = 2
        self.grid[self._goal_loc] = 3

    def world_coords_to_grid_index(self, coord: np.ndarray) -> tuple[int, int]:
        """
        Convert world coordinates (as [x,y]) into grid indices (row,col) of the cell containing that location

        The grid locs are oriented such that the top left of the image is
        0,0, and the bottom right is (max_y, max_x), so that if we look at the array
        it is the same in layout as the xy coordinate plot

        In other words, the column index increases in -y, and the row index increases
        in +x

        :param coord: world coordinate in [x,y]
        :type coord: np.ndarray
        :return: grid indices [row, col]
        :rtype: tuple[int, int]
        """
        # todo check for invalid input

        row = np.floor((self.c.dimensions[1, 1] - coord[1]) / self.c.cell_size).astype(
            int
        )
        col = np.floor((coord[0] - self.c.dimensions[0, 0]) / self.c.cell_size).astype(
            int
        )
        return (row, col)

    def grid_index_to_world_coords_corner(
        self, loc: tuple[int, int] | np.ndarray
    ) -> np.ndarray:
        """
        Convert grid index (row, col) of a cell into the world coordinates
        of that cell's top left corner.

        :param loc: grid index (row, col)
        :type loc: tuple[int, int] | np.ndarray
        :return: world coordinate (x,y)
        """
        x = self.c.dimensions[0, 0] + (loc[1] * self.c.cell_size)
        y = self.c.dimensions[1, 1] - (loc[0] * self.c.cell_size)
        return np.array((x, y))

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

    def get_obstacles(self) -> list[np.ndarray]:
        return self._obstacles

    def get_start_loc(self) -> tuple[int, int]:
        return self._start_loc

    def get_goal_loc(self) -> tuple[int, int]:
        return self._goal_loc

    def get_neighbors(self, loc: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Get all unoccupied neighbors of grid location "loc". This means they're on the grid,
        and they don't contain an obstacle
        """
        neighbors = []
        for xdiff in (-1, 1):
            for ydiff in (-1, 1):
                try:
                    neighbor = (loc[0] + xdiff, loc[1] + ydiff)
                    # if it's out of range, we get an index error here:
                    val = self.grid[neighbor]

                    if val != 1:
                        neighbors.append(neighbor)
                except IndexError:
                    continue
        return neighbors

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
