"""
"online" version of A*

in this case, this just means that I run offline A* once every step
& hide the obstacles until i'm close to them.
"""

from copy import copy

import numpy as np
from hw1.astar import AStar, Node, Path
from hw1.data import Dataset
from hw1.map import Map


def run_astar_online(ds: Dataset, cfg: Map.Config) -> tuple[Path, Map]:
    """
    Run A* "online", such that:
    - we can only see obstacles when we are adjacent to them, and start out
      with no knowledge of any of them

    :param ds: dataset containing obstacles
    :param cfg: map configuration
    :return: the Path the robot took, and the Map it discovered
    """

    robot_map_cfg = copy(cfg)
    # the actual robot map has an obstacle radius of 0.
    # it just checks against a precomputed ground-truth map with the
    # correct obstacle radius.
    robot_map_cfg.obstacle_radius = 0.0
    map = Map(robot_map_cfg, [])
    algo = AStar()
    path = Path()

    # ground-truth obstacle representations
    all_obstacles = [
        np.array((x, y)) for x, y in zip(ds.landmarks["x_m"], ds.landmarks["y_m"])
    ]
    gt_map = Map(cfg, all_obstacles)

    robot_loc = map.get_start_loc()
    path.nodes.append(Node(loc=robot_loc))

    while robot_loc != map.get_goal_loc():
        # add any obstacles we can now see to the map (all neighboring obstacles
        # in the ground-truth grid.)
        obs_locs = gt_map.get_neighbors(robot_loc, return_obstacles=True)
        map.add_obstacle_locs(obs_locs)

        # plan a path given our current knowledge
        p = algo.solve(map, robot_loc)

        # move to the next location in the new path
        robot_loc = p.locs[1]
        path.nodes.append(p.nodes[1])

    return path, map


def run_astar_control_online(ds: Dataset, cfg: Map.Config) -> tuple[Path, Map]:
    """
    Run A* "online", such that:
    - we can only see obstacles when we are adjacent to them, and start out
      with no knowledge of any of them

    :param ds: dataset containing obstacles
    :param cfg: map configuration
    :return: the Path the robot took, and the Map it discovered
    """

    robot_map_cfg = copy(cfg)
    # the actual robot map has an obstacle radius of 0.
    # it just checks against a precomputed ground-truth map with the
    # correct obstacle radius.
    robot_map_cfg.obstacle_radius = 0.0
    map = Map(robot_map_cfg, [])
    algo = AStar()
    path = Path()

    # ground-truth obstacle representations
    all_obstacles = [
        np.array((x, y)) for x, y in zip(ds.landmarks["x_m"], ds.landmarks["y_m"])
    ]
    gt_map = Map(cfg, all_obstacles)

    robot_loc = map.get_start_loc()
    path.nodes.append(Node(loc=robot_loc))

    while robot_loc != map.get_goal_loc():
        # add any obstacles we can now see to the map (all neighboring obstacles
        # in the ground-truth grid.)
        obs_locs = gt_map.get_neighbors(robot_loc, return_obstacles=True)
        map.add_obstacle_locs(obs_locs)

        # plan a path given our current knowledge
        p = algo.solve(map, robot_loc)

        # move to the next location in the new path
        robot_loc = p.locs[1]
        path.nodes.append(p.nodes[1])

    return path, map
