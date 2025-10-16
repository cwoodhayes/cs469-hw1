"""
"online" version of A*

in this case, this just means that I run offline A* once every step
& hide the obstacles until i'm close to them.
"""

import numpy as np
from hw0.astar import AStar, Node, Path
from hw0.data import Dataset
from hw0.map import Map, WorldObstacles


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
