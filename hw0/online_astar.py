"""
"online" version of A*

in this case, this just means that I run offline A* once every step
& hide the obstacles until i'm close to them.
"""

import numpy as np
from hw0.astar import AStar, Node, Path
from hw0.data import Dataset
from hw0.map import Map


def run_astar_online(
    ds: Dataset, cfg: Map.Config, obstacle_radius: float | None = None
) -> tuple[Path, Map]:
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
    obstacles = [
        np.array((x, y)) for x, y in zip(ds.landmarks["x_m"], ds.landmarks["y_m"])
    ]
    obs_loc_to_coords = {map.world_coords_to_grid_loc(obs): obs for obs in obstacles}
    obs_locs = set(obs_loc_to_coords.keys())

    robot_loc = map.get_start_loc()
    path.nodes.append(Node(loc=robot_loc))

    while robot_loc != map.get_goal_loc():
        # add any obstacles we can see to the map
        neighbor_locs = map.get_neighbors(robot_loc)
        adjacent_obs_locs = neighbor_locs.intersection(obs_locs)
        obs_coords = [obs_loc_to_coords[loc] for loc in adjacent_obs_locs]
        map.add_obstacles(obs_coords)

        # plan a path given our current knowledge
        p = algo.solve(map, robot_loc)

        # move to the next location in the new path
        robot_loc = p.locs[1]
        path.nodes.append(p.nodes[1])

    return path, map
