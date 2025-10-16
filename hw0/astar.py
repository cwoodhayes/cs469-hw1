"""
"Offline" A* solver module. Given a grid map, a goal G, and a start S, plan a path
through the grid.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
import heapq

import numpy as np

from hw0.map import Map


@dataclass(order=True)
class Node:
    """Node in the A* search. contains a priority value with the grid location"""

    loc: tuple[int, int] = field(compare=False)
    f_score: float = float("inf")

    def __hash__(self) -> int:
        # we can only have one copy of each location. f_score doesn't matter.
        return hash(self.loc)


@dataclass
class Path:
    """
    Represents a path through the map
    """

    nodes: deque[Node] = field(default_factory=deque)
    """nodes in a path from start to goal. uses deque cuz i have to prepend"""

    @property
    def locs(self) -> list[np.ndarray]:
        return [np.array(n.loc) for n in self.nodes]


class AStar:
    """
    "Offline" A* solver. Given a grid map, a goal G, and a start S, plan a path
    through the grid.

    CITATION:
    https://en.wikipedia.org/wiki/A*_search_algorithm

    ^ I referred to the pseudocode here in writing my implementation
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def heuristic(map: Map, loc: tuple[int, int]) -> float:
        """
        Gets a grid location (node) and the map, and outputs a scalar heuristic value.

        for now, we use the simplest, easiest to calculate admissible heuristic:
        euclidian distance to the goal divided by the cell size
        """
        goal = map.get_goal_loc()
        dist = ((goal[0] - loc[0]) ** 2 + (goal[1] - loc[1]) ** 2) ** 0.5
        return dist

    @staticmethod
    def reconstruct_path(came_from: dict[Node, Node], current: Node) -> Path:
        p = Path()
        p.nodes.appendleft(current)
        while current in came_from.keys():
            current = came_from[current]
            p.nodes.appendleft(current)

        return p

    def solve(self, map: Map) -> Path:
        """
        Given a grid map, a goal G, and a start S, plan a path
        through the grid.

        :param map: map of the world from the robot's perspective
        :return: path from S to G
        """
        # start & end nodes
        gloc = map.get_goal_loc()
        sloc = map.get_start_loc()
        start = Node(
            loc=sloc,
            f_score=self.heuristic(map, sloc),
        )

        # various maps & priority queues and such
        openset: list[Node] = [start]
        came_from: dict[Node, Node] = {}
        g_score: defaultdict[Node, float] = defaultdict(lambda: float("inf"))
        g_score[start] = 0

        f_score: defaultdict[Node, float] = defaultdict(lambda: float("inf"))
        f_score[start] = start.f_score

        while len(openset) > 0:
            current = heapq.heappop(openset)
            print(f"CURRENT: {current}")

            if current.loc == gloc:
                return self.reconstruct_path(came_from, current)

            neighbor_locs = map.get_neighbors(current.loc)
            for neighbor_loc in neighbor_locs:
                # our edge weights are all 1, since get_neighbors ignores obstacles.
                new_gscore = g_score[current] + 1
                n = Node(neighbor_loc)
                # print(f"NEIGHBOR: {n}")
                if new_gscore < g_score[n]:
                    print(f"ADDING: {n}")
                    # this path to n is better than any previous one. record it
                    came_from[n] = current
                    g_score[n] = new_gscore
                    n.f_score = new_gscore + self.heuristic(map, n.loc)
                    f_score[n] = n.f_score
                    if n not in openset:
                        heapq.heappush(openset, n)

        # failure--there's no path to the goal
        return Path()
