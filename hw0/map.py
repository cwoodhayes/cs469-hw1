"""
Map of the world according to the robot, represented in world coordinates
"""

import numpy as np

from hw0.data import Dataset


class Map:
    """
    Map of the world in a grid form
    """

    def __init__(self, map_dimensions: np.ndarray, cell_size: float, obstacles: list[np.ndarray], start: np.ndarray, goal: np.ndarray) -> None:
        """
        :param map_dimensions: dimensions of the map in meters [[xmin, xmax], [ymin, ymax]]
        :type map_dimensions: np.ndarray[[xlim],[ylim]]
        :param cell_size: side size of an individual square cell in meters
        :type cell_size: tuple[float, float]
        :param obstacles: Nx2 array of obstacles, where each row is an (x,y) location of an obstacle in world coords
        :param start: start location in world coords (x,y)
        :param goal: goal location in world coords (x,y)
        """
        dim = np.array(map_dimensions)
        self._dim = dim
        self._world_mins = dim[:,0]
        self._cell_size = cell_size
        x_range = dim[0,1] - dim[0,0]
        y_range = dim[1,1] - dim[1,0]
        range = np.array([y_range, x_range])

        shape = range / cell_size
        shape_int = shape.round().astype(int)
        if not np.isclose(shape_int, shape).all():
            raise ValueError(f"{dim} grid does not break evenly into cells of size {cell_size}")

        self.grid = np.zeros(shape=shape_int, dtype=int)

        self.add_obstacles(obstacles)
        start_loc = self.world_coords_to_grid_index(start)
        goal_loc = self.world_coords_to_grid_index(goal)
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
    
    @classmethod
    def construct_from_dataset(cls, ds: Dataset) -> None:
        """
        construct a map with all the obstacles from the dataset
        
        :param ds: Dataset object
        :type ds: Dataset
        """