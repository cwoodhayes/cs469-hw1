"""
plotting functions for robot data
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib import patches
from matplotlib.ticker import MultipleLocator

from hw1.map import Map
from hw1 import astar


def plot_map(map: Map, ax: Axes, groundtruth_map: Map | None = None) -> None:
    ##### Plot the obstacles
    patch = None
    obs_locs = map.get_obstacle_locs()
    for loc in obs_locs:
        obs = map.grid_loc_to_world_coords_corner(loc)
        patch = patches.Rectangle(
            obs,  # type: ignore
            map.c.cell_size,
            -map.c.cell_size,
            facecolor="black",
        )
        ax.add_patch(patch)
    if patch is not None:
        patch.set_label("Obstacle")

    ### If supplied, plot ground-truth obstacles behind those in light grey
    if groundtruth_map is not None:
        patch = None
        obs_locs = groundtruth_map.get_obstacle_locs()
        for loc in obs_locs:
            obs = map.grid_loc_to_world_coords_corner(loc)
            patch = patches.Rectangle(
                obs,  # type: ignore
                map.c.cell_size,
                -map.c.cell_size,
                facecolor="#00000037",
            )
            ax.add_patch(patch)
        if patch is not None:
            patch.set_label("Undiscovered Obstacle")

    ##### Plot start and goal
    goal_corner = map.grid_loc_to_world_coords_corner(map._goal_loc)
    start_corner = map.grid_loc_to_world_coords_corner(map._start_loc)
    ax.add_patch(
        patches.Rectangle(
            goal_corner,  # type: ignore
            map.c.cell_size,
            -map.c.cell_size,
            color="#FFD90099",
            label="Goal",
        )
    )
    ax.add_patch(
        patches.Rectangle(
            start_corner,  # type: ignore
            map.c.cell_size,
            -map.c.cell_size,
            color="#00FF5599",
            label="Start",
        )
    )

    ##### Make legend and grid
    xlim = map.c.dimensions[0, :]
    ylim = map.c.dimensions[1, :]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.grid(True)
    ax.xaxis.set_minor_locator(MultipleLocator(map.c.cell_size))
    ax.yaxis.set_minor_locator(MultipleLocator(map.c.cell_size))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))


def plot_path_on_map(
    map: Map,
    ax: Axes,
    p: astar.Path,
    groundtruth_map: Map | None = None,
    plot_centers: bool = True,
) -> None:
    """
    plots the robot path discoverd by A* on the map.

    if supplied, groundtruth_map supplies the obstacles that were
    not discovered by the robot, which are displayed in grey
    """
    plot_map(map, ax, groundtruth_map)

    # Fill in every cell visited in light blue
    centers = []
    rect = None
    for loc in p.locs:
        corner = map.grid_loc_to_world_coords_corner(loc)
        rect = patches.Rectangle(
            corner,  # type: ignore
            map.c.cell_size,
            -map.c.cell_size,
            color="#4590E57B",
        )
        ax.add_patch(rect)
        center_x = corner[0] + map.c.cell_size / 2
        center_y = corner[1] - map.c.cell_size / 2
        centers.append((center_x, center_y))

    if rect is not None:
        rect.set_label("Robot Path")

    c_arr = np.array(centers)

    if plot_centers:
        ax.plot(c_arr[:, 0], c_arr[:, 1], "bo-", ms=4, label="Robot Path")

    # add a single dot for start and goal colors (for when the grid cell is small)
    ax.plot(c_arr[0, 0], c_arr[0, 1], marker="o", color="#00FF55", ms=10, zorder=1.1)
    ax.plot(c_arr[-1, 0], c_arr[-1, 1], marker="o", color="#FFD900", ms=10, zorder=1.1)


def plot_trajectory_over_waypoints(
    ax: Axes,
    traj: np.ndarray,
    waypoints: np.ndarray,
    distance_threshold: float,
) -> None:
    """
    Plot a controlled robot trajectory

    and the waypoints it's attempting to reach.

    :param ax: plt axes
    :param traj: trajectory: [[x, y], ...]
    :param waypoints: list of control target waypoints [[x, y], ...]
    :param distance_threshold: "close enough" radius used to evaluate
        whether a waypoint was reached
    """

    ax.scatter(waypoints[:, 0], waypoints[:, 1], c="#BB4C4C", label="Waypoint")
    ax.plot(traj[:, 0], traj[:, 1], "bo-", ms=4, label="Robot Path")

    c = None
    for wp in waypoints:
        c = patches.Circle(
            wp, distance_threshold, edgecolor="#4F8FF6", facecolor=(1, 1, 1, 0)
        )
        ax.add_patch(c)
    if c is not None:
        c.set_label("Waypoint radius")

    ax.legend()
