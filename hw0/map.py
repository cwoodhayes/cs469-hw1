"""
Map of the world according to the robot, represented in world coordinates
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hw0.data import Dataset
from hw0.utils import write_square_kernel_with_clip


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

        obstacle_radius: float = 0
        # radius is only relevant for q7 and above

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
        # maintain these lists mostly for plotting
        self._obstacles = list()
        self._obstacle_radius_idx = round(self.c.obstacle_radius / self.c.cell_size)

        self.add_obstacles(obstacles)
        self._start_loc = self.world_coords_to_grid_loc(config.start)
        self._goal_loc = self.world_coords_to_grid_loc(config.goal)
        self.grid[self._start_loc] = 2
        self.grid[self._goal_loc] = 3

    def world_coords_to_grid_loc(self, coord: np.ndarray) -> tuple[int, int]:
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

    def grid_loc_to_world_coords_corner(
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

    def world_coords_to_corner(self, coord: np.ndarray) -> np.ndarray:
        """
        convenience method for plotting. returns the top left corner of the cell
        containing "coord", in world coordinates.
        """
        loc = self.world_coords_to_grid_loc(coord)
        return self.grid_loc_to_world_coords_corner(loc)

    def add_obstacles(self, obstacles: Iterable[np.ndarray]) -> None:
        """
        Add a set of obstacles to the map.
        (we only ever add them, we never remove)

        :param obstacles: Nx2 list of obstacles, where each row is an (x,y) location of an obstacle in world coords
        """
        for obs in obstacles:
            row, col = self.world_coords_to_grid_loc(obs)
            write_square_kernel_with_clip(
                self.grid, (row, col), self._obstacle_radius_idx, 1
            )

            self._obstacles.append(obs)

    def get_obstacles(self) -> list[np.ndarray]:
        """
        Returns all obstacle anchor points (centers) in world-coordinates
        """
        return self._obstacles

    def get_obstacle_locs(self) -> np.ndarray:
        """
        :return: Nx2 array of grid locations.
        """
        return np.argwhere(self.grid == 1)

    def get_start_loc(self) -> tuple[int, int]:
        return self._start_loc

    def get_goal_loc(self) -> tuple[int, int]:
        return self._goal_loc

    def get_neighbors(self, loc: tuple[int, int]) -> set[tuple[int, int]]:
        """
        Get all unoccupied neighbors of grid location "loc". This means they're on the grid,
        and they don't contain an obstacle
        """
        neighbors = set()
        for xdiff in (-1, 0, 1):
            for ydiff in (-1, 0, 1):
                neighbor = (loc[0] + xdiff, loc[1] + ydiff)
                if neighbor == loc:
                    continue
                if neighbor[0] < 0 or neighbor[0] >= self.grid.shape[0]:
                    continue
                if neighbor[1] < 0 or neighbor[1] >= self.grid.shape[1]:  # type: ignore
                    continue
                val = self.grid[neighbor]
                if val != 1:
                    neighbors.add(neighbor)
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


class WorldObstacles:
    def __init__(self, all_obstacles: list[np.ndarray]) -> None:
        self._obstacles = all_obstacles

    def obstacles_within_radius(
        self, radius: float, my_coord: np.ndarray
    ) -> list[np.ndarray]:
        """
        Find all obstacles within a given radius of a world coordinate

        :param radius: distance in meters
        :param my_coord: robot location in world coordinates
        :rtype: subset of the obstacle list within the radius
        """
        # this is a good place to optimize if runtime is slow
        # not overthinking it for the first pass
        out = [
            obs_coord
            for obs_coord in self._obstacles
            if np.linalg.norm(my_coord - obs_coord) <= radius
        ]
        return out

    def obstacles_within_radius_loc(
        self, map: Map, radius: float, my_loc: np.ndarray
    ) -> list[np.ndarray]:
        """
        Find all obstacles within a radius of a given grid location.
        Measures from the center of the grid cell
        """
        # measure from the center of the grid cell
        corner = map.grid_loc_to_world_coords_corner(my_loc)
        dist = map.c.cell_size / 2
        center = np.array(((corner[0] + dist), (corner[1] - dist)))
        return self.obstacles_within_radius(radius, center)
