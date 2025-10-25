"""
author: conor hayes
"""

import pathlib
import signal

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from hw1.astar import AStar
from hw1.data import Dataset
from hw1.map import Map
from hw1.plot import plot_path_on_map, plot_trajectory_over_waypoints
from hw1.online_astar import run_astar_online
from hw1.motion_control import WaypointController, RobotNavSim


REPO_ROOT = pathlib.Path(__file__).parent


def main():
    print("cs469 Homework 1")
    # make matplotlib responsive to ctrl+c
    # cite: this stackoverflow answer:
    # https://stackoverflow.com/questions/67977761/how-to-make-plt-show-responsive-to-ctrl-c
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # my assigned dataset is ds1, so I'm hardcoding this
    ds = Dataset.from_dataset_directory(REPO_ROOT / "data/ds1")

    # q3(ds)
    # q5(ds)
    # q7(ds)
    q8(ds)
    # q9(ds)
    q10(ds)

    plt.show()


def q10(ds: Dataset):
    starts = [(2.45, -3.55), (4.95, -0.05), (-0.55, 1.45)]
    goals = [(0.95, -1.55), (2.45, 0.25), (1.95, 3.95)]

    fig = plt.figure(figsize=(20, 12))
    axes: list[Axes] = fig.subplots(1, 3)

    ctl_cfg = WaypointController.Config(
        vK_p=0.1,
        vp_0=0.03,
        wK_p=6.0,
        wp_0=0.0,
        vdot_max=0.288,
        wdot_max=5.579,
    )
    sim = RobotNavSim(RobotNavSim.Config(), WaypointController(ctl_cfg))

    for start_loc, goal_loc, idx in zip(starts, goals, range(3)):
        cfg = Map.Config(
            dimensions=np.array(
                [
                    [-2, 5],
                    [-6, 6],
                ]
            ),
            cell_size=0.1,
            start=np.array(start_loc),
            goal=np.array(goal_loc),
            obstacle_radius=0.3,
        )

        path, map, traj = run_astar_online(ds, cfg, sim)
        path.print()

        waypoints = np.array(path.get_centers(map))
        groundtruth_map = Map.construct_from_dataset(ds, cfg)

        plot_path_on_map(map, axes[idx], path, groundtruth_map, plot_centers=False)
        plot_trajectory_over_waypoints(axes[idx], traj, waypoints, sim.c.dist_thresh_m)
        total_t = sim.c.dt * len(traj)
        axes[idx].set_title(f"S={start_loc}, G={goal_loc}, t={total_t}s")

    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.suptitle("Q10: Online A*, online control", fontsize=16, fontweight="bold")
    fig.show()


def q9(ds: Dataset):
    starts = [(2.45, -3.55), (4.95, -0.05), (-0.55, 1.45)]
    goals = [(0.95, -1.55), (2.45, 0.25), (1.95, 3.95)]

    fig = plt.figure(figsize=(20, 12))
    axes: list[Axes] = fig.subplots(1, 3)

    for start_loc, goal_loc, idx in zip(starts, goals, range(3)):
        cfg = Map.Config(
            dimensions=np.array(
                [
                    [-2, 5],
                    [-6, 6],
                ]
            ),
            cell_size=0.1,
            start=np.array(start_loc),
            goal=np.array(goal_loc),
            obstacle_radius=0.3,
        )

        path, map, _ = run_astar_online(ds, cfg)
        path.print()

        # target the waypoints above
        waypoints = np.array(path.get_centers(map))
        # ignore the first waypoint--that's our start
        x = np.array([*waypoints[0], -np.pi / 2])
        waypoints = waypoints[1:, :]
        ctl_cfg = WaypointController.Config(
            vK_p=0.1,
            vp_0=0.03,
            wK_p=6.0,
            wp_0=0.0,
            vdot_max=0.288,
            wdot_max=5.579,
        )
        sim = RobotNavSim(RobotNavSim.Config(), WaypointController(ctl_cfg))
        all_x = []
        u = np.full((2,), np.nan, dtype=np.float32)

        for wp_idx in range(len(waypoints)):
            traj, u_all = sim.navigate(x, u, waypoints[wp_idx])
            x = traj[-1]
            u = u_all[-1] if len(u_all) > 0 else u
            print(f"found waypoint {wp_idx} (it={len(traj)})")
            all_x.extend(traj)

        groundtruth_map = Map.construct_from_dataset(ds, cfg)
        plot_path_on_map(map, axes[idx], path, groundtruth_map, plot_centers=False)
        plot_trajectory_over_waypoints(
            axes[idx], np.array(all_x), waypoints, sim.c.dist_thresh_m
        )
        axes[idx].set_title(f"S={start_loc}, G={goal_loc}")

    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.suptitle("Q9: Online A*, post-hoc control", fontsize=16, fontweight="bold")
    fig.show()


def q8(ds: Dataset) -> None:
    # simulate our controller navigating to some sample points
    print("Part B, Question 8:")
    waypoints = np.array(
        [
            [0, 0],
            [5, 5],
            [10, 0],
            [12, 9],
            [0, 3],
        ]
    )
    stddevs = [0.0, 0.1, 0.4, 0.8]
    dt = 0.1
    fig = plt.figure(figsize=(20, 6))
    axes = fig.subplots(1, 4)

    for std, ax in zip(stddevs, axes):
        # target the waypoints above
        sim_cfg = RobotNavSim.Config(dt=dt, x_noise_stddev=std, dist_thresh_m=0.5)
        sim = RobotNavSim(sim_cfg, WaypointController())
        all_x = []
        x = np.array([0.0, 0.0, 0.0])
        u = np.full((2,), np.nan, dtype=np.float32)

        for wp_idx in range(len(waypoints)):
            traj, u_all = sim.navigate(x, u, waypoints[wp_idx])
            x = traj[-1]
            u = u_all[-1] if len(u_all) > 0 else u
            print(f"found waypoint {wp_idx} (it={len(traj)})")
            all_x.extend(traj)

        plot_trajectory_over_waypoints(
            ax, np.array(all_x), waypoints, sim.c.dist_thresh_m
        )
        ax.set_title(f"x noise stddev={std} (#iter={len(all_x)})")

    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.suptitle(
        f"Q8: Motion control for 5 waypoints, dt={dt}",
        fontsize=16,
        fontweight="bold",
    )
    fig.show()


def q7(ds: Dataset):
    print("Part A, Question 7:")

    starts = [(2.45, -3.55), (4.95, -0.05), (-0.55, 1.45)]
    goals = [(0.95, -1.55), (2.45, 0.25), (1.95, 3.95)]

    fig = plt.figure(figsize=(10, 6))
    axes: list[Axes] = fig.subplots(1, 3)

    for start_loc, goal_loc, idx in zip(starts, goals, range(3)):
        cfg = Map.Config(
            dimensions=np.array(
                [
                    [-2, 5],
                    [-6, 6],
                ]
            ),
            cell_size=0.1,
            start=np.array(start_loc),
            goal=np.array(goal_loc),
            obstacle_radius=0.3,
        )

        path, map, _ = run_astar_online(ds, cfg)
        path.print()

        groundtruth_map = Map.construct_from_dataset(ds, cfg)
        plot_path_on_map(map, axes[idx], path, groundtruth_map, plot_centers=False)
        axes[idx].set_title(f"S={start_loc}, G={goal_loc}")

    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.suptitle(
        "Q7: Online A* paths (cell size = .1x.1m)", fontsize=16, fontweight="bold"
    )
    fig.show()


def q5(ds: Dataset):
    print("Part A, Question 5:")

    starts = [(0.5, -1.5), (4.5, 3.5), (-0.5, 5.5)]
    goals = [(0.5, 1.5), (4.5, -1.5), (1.5, -3.5)]

    fig = plt.figure(figsize=(10, 6))
    axes: list[Axes] = fig.subplots(1, 3)

    for start_loc, goal_loc, idx in zip(starts, goals, range(3)):
        cfg = Map.Config(
            dimensions=np.array(
                [
                    [-2, 5],
                    [-6, 6],
                ]
            ),
            cell_size=1.0,
            start=np.array(start_loc),
            goal=np.array(goal_loc),
        )

        path, map, _ = run_astar_online(ds, cfg)
        path.print()

        groundtruth_map = Map.construct_from_dataset(ds, cfg)
        plot_path_on_map(map, axes[idx], path, groundtruth_map)
        axes[idx].set_title(f"S={start_loc}, G={goal_loc}")

    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.suptitle("Q5: Online A* paths", fontsize=16, fontweight="bold")
    fig.show()


def q3(ds: Dataset):
    print("Part A, Question 3:")

    starts = [(0.5, -1.5), (4.5, 3.5), (-0.5, 5.5)]
    goals = [(0.5, 1.5), (4.5, -1.5), (1.5, -3.5)]

    fig = plt.figure(figsize=(10, 6))
    axes: list[Axes] = fig.subplots(1, 3)

    for start_loc, goal_loc, idx in zip(starts, goals, range(3)):
        cfg = Map.Config(
            dimensions=np.array(
                [
                    [-2, 5],
                    [-6, 6],
                ]
            ),
            cell_size=1.0,
            start=np.array(start_loc),
            goal=np.array(goal_loc),
        )

        map = Map.construct_from_dataset(ds, cfg)

        algo = AStar()
        path = algo.solve(map)
        path.print()

        plot_path_on_map(map, axes[idx], path)
        axes[idx].set_title(f"S={start_loc}, G={goal_loc}")

    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.subplots_adjust(bottom=0.15)

    fig.suptitle("Q3: Offline A* solutions", fontsize=16, fontweight="bold")
    fig.show()


if __name__ == "__main__":
    main()
