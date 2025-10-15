"""
"Offline" A* solver module. Given a grid map, a goal G, and a start S, plan a path
through the grid. 
"""

from dataclasses import dataclass

import numpy as np

from hw0.map import Map


class AStar():
    """
    "Offline" A* solver. Given a grid map, a goal G, and a start S, plan a path
    through the grid. 
    """

    def __init__(self) -> None:
        pass

    def solve(self, map: Map):
        """
        Given a grid map, a goal G, and a start S, plan a path
        through the grid. 
        """