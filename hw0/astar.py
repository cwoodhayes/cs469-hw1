"""
"Offline" A* solver module. Given a grid map, a goal G, and a start S, plan a path
through the grid.
"""

from dataclasses import dataclass, field

import numpy as np

from hw0.map import Map


@dataclass
class Path:
    """
    Represents a path through the map
    """

    locs: list[np.ndarray] = field(default_factory=list)
    """2xN array of N cell locations in a path from start to goal"""


class AStar:
    """
    "Offline" A* solver. Given a grid map, a goal G, and a start S, plan a path
    through the grid.
    """

    def __init__(self) -> None:
        pass

    def solve(self, map: Map) -> Path:
        """
        Given a grid map, a goal G, and a start S, plan a path
        through the grid.

        :param map: map of the world from the robot's perspective
        :return:
        """
        # TODO fill this in

        # for now just return dummy path to bottom of map
        p = Path()
        for row_idx in range(map.get_start_loc()[0], map.grid.shape[0]):
            p.locs.append(np.array((row_idx, map.get_start_loc()[1])))

        return p
