"""
"online" version of A*

in this case, this just means that I run offline A* once every step
& hide the obstacles until i'm close to them.
"""

import numpy as np
from hw0.astar import AStar, Node, Path
from hw0.data import Dataset
from hw0.map import Map


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
        center = np.array((corner[0] + dist), (corner[1] - dist))
        return self.obstacles_within_radius(radius, center)


def run_astar_online(ds: Dataset, cfg: Map.Config) -> tuple[Path, Map]:
    """
    Run A* "online", such that:
    - we can only see obstacles when we are adjacent to them, and start out
      with no knowledge of any of them

    :param ds: dataset containing obstacles
    :param cfg: map configuration
    :return: the Path the robot took, and the Map it discovered
    """

    map = Map(cfg, [])
    algo = AStar()
    path = Path()

    # ground-truth obstacle representations
    all_obstacles = [
        np.array((x, y)) for x, y in zip(ds.landmarks["x_m"], ds.landmarks["y_m"])
    ]
    world_obs = WorldObstacles(all_obstacles)

    robot_loc = map.get_start_loc()
    path.nodes.append(Node(loc=robot_loc))

    while robot_loc != map.get_goal_loc():
        # add any obstacles we can see to the map
        obs_coords = world_obs.obstacles_within_radius_loc(
            map, map.c.obstacle_radius + map.c.cell_size, robot_loc
        )
        map.add_obstacles(obs_coords)

        # plan a path given our current knowledge
        p = algo.solve(map, robot_loc)

        # move to the next location in the new path
        robot_loc = p.locs[1]
        path.nodes.append(p.nodes[1])

    return path, map
