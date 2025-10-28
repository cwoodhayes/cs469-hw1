"""
Entry point.

Simulates A* plus a low level controller for path planning & execution.

author: conor hayes
"""

import argparse
import pathlib
import signal

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from hw1.astar import AStar
from hw1.data import Dataset
from hw1.map import Map
from hw1.plot import plot_path_on_map, plot_trajectory_over_waypoints
from hw1.online_astar import run_astar_online
from hw1.motion_control import WaypointController, RobotNavSim


REPO_ROOT = pathlib.Path(__file__).parent
FIGURES_DIR = REPO_ROOT / "figures"


def main():
    print("cs469 Homework 1")
    # make matplotlib responsive to ctrl+c
    # cite: this stackoverflow answer:
    # https://stackoverflow.com/questions/67977761/how-to-make-plt-show-responsive-to-ctrl-c
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    plt.rcParams["legend.fontsize"] = 14

    ns = get_cli_args()

    # my assigned dataset is ds1, so I'm hardcoding this
    ds = Dataset.from_dataset_directory(REPO_ROOT / "data/ds1")

    q3(ds)
    q5(ds)
    q7(ds)
    q8(ds)
    q9(ds)
    q10(ds)
    q11(ds)

    if ns.save:
        print("Saving figures...")
        for num in plt.get_fignums():
            fig = plt.figure(num)
            name = fig.get_label() or f"figure_{num}"
            fig.savefig(FIGURES_DIR / f"{name}.png")
    else:
        plt.show()


def get_cli_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser("online A* + diff-drive control simulator")
    cli.add_argument(
        "-s",
        "--save",
        action="store_true",
    )
    return cli.parse_args()


def q11(ds: Dataset):
    starts = [(0.5, -1.5), (4.5, 3.5), (-0.5, 5.5)]
    goals = [(0.5, 1.5), (4.5, -1.5), (1.5, -3.5)]

    u_noise = 0.1

    ctl_cfg = WaypointController.Config(
        vK_p=0.1,
        vp_0=0.03,
        wK_p=6.0,
        wp_0=0.0,
        vdot_max=0.288,
        wdot_max=5.579,
    )
    sim = RobotNavSim(
        RobotNavSim.Config(
            w_noise_stddev_percent_wmax=u_noise,
            v_noise_stddev_percent_vmax=u_noise,
            x_noise_stddev=0.0,
        ),
        WaypointController(ctl_cfg),
    )

    fig = plt.figure("Q11", figsize=(20, 13))
    axes: list[Axes] = fig.subplots(1, 3)

    for start_loc, goal_loc, idx in zip(starts, goals, range(3)):
        for cell_size in [0.1, 1.0]:
            cfg = Map.Config(
                dimensions=np.array(
                    [
                        [-2, 5],
                        [-6, 6],
                    ]
                ),
                cell_size=cell_size,
                start=np.array(start_loc),
                goal=np.array(goal_loc),
                obstacle_radius=0.3,
            )

            path, map, traj = run_astar_online(ds, cfg, sim)
            path.print()

            waypoints = np.array(path.get_centers(map))
            groundtruth_map = Map.construct_from_dataset(ds, cfg)

            superimpose = True if cell_size == 1.0 else False
            plot_path_on_map(
                map,
                axes[idx],
                path,
                groundtruth_map,
                plot_centers=False,
                show_full_map=True,
                is_superimposed=superimpose,
            )
            plot_trajectory_over_waypoints(
                axes[idx],
                traj,
                waypoints,
                sim.c.dist_thresh_m,
                secondary_trajectory=superimpose,
            )
            total_t = round(sim.c.dt * len(traj), 2)
            axes[idx].set_title(
                f"S={start_loc}, G={goal_loc} (#iter={len(traj)}, t={total_t}s)",
                fontsize=16,
            )

    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.subplots_adjust(bottom=0.15)
    fig.suptitle(
        "Q11: Online A*, online control (varying cell size)",
        fontsize=22,
        fontweight="bold",
    )


def q10(ds: Dataset):
    starts = [(2.45, -3.55), (4.95, -0.05), (-0.55, 1.45)]
    goals = [(0.95, -1.55), (2.45, 0.25), (1.95, 3.95)]

    u_noise = 0.2
    fig = run_sim(ds, starts, goals, 0.1, "Q10", False, u_noise=u_noise)
    fig.suptitle(
        "Q10: Online A*, online control (v, w noise stddev=0.2)",
        fontsize=16,
        fontweight="bold",
    )
    fig.show()


def run_sim(
    ds: Dataset,
    starts: list[tuple],
    goals: list[tuple],
    cell_size: float,
    figname: str,
    show_full_map: bool = False,
    x_noise: float = 0.0,
    u_noise: float = 0.2,
) -> Figure:
    fig = plt.figure(figname, figsize=(20, 13))
    axes: list[Axes] = fig.subplots(1, 3)

    ctl_cfg = WaypointController.Config(
        vK_p=0.1,
        vp_0=0.03,
        wK_p=6.0,
        wp_0=0.0,
        vdot_max=0.288,
        wdot_max=5.579,
    )
    sim = RobotNavSim(
        RobotNavSim.Config(
            w_noise_stddev_percent_wmax=u_noise,
            v_noise_stddev_percent_vmax=u_noise,
            x_noise_stddev=x_noise,
        ),
        WaypointController(ctl_cfg),
    )

    for start_loc, goal_loc, idx in zip(starts, goals, range(3)):
        cfg = Map.Config(
            dimensions=np.array(
                [
                    [-2, 5],
                    [-6, 6],
                ]
            ),
            cell_size=cell_size,
            start=np.array(start_loc),
            goal=np.array(goal_loc),
            obstacle_radius=0.3,
        )

        path, map, traj = run_astar_online(ds, cfg, sim)
        path.print()

        waypoints = np.array(path.get_centers(map))
        groundtruth_map = Map.construct_from_dataset(ds, cfg)

        plot_path_on_map(
            map,
            axes[idx],
            path,
            groundtruth_map,
            plot_centers=False,
            show_full_map=show_full_map,
        )
        plot_trajectory_over_waypoints(axes[idx], traj, waypoints, sim.c.dist_thresh_m)
        total_t = round(sim.c.dt * len(traj), 2)
        axes[idx].set_title(
            f"S={start_loc}, G={goal_loc} (#iter={len(traj)}, t={total_t}s)"
        )

    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.subplots_adjust(bottom=0.15)
    return fig


def q9(ds: Dataset):
    starts = [(2.45, -3.55), (4.95, -0.05), (-0.55, 1.45)]
    goals = [(0.95, -1.55), (2.45, 0.25), (1.95, 3.95)]

    stddev = 0.02
    fig = plt.figure("Q9", figsize=(20, 12))
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
        sim = RobotNavSim(
            RobotNavSim.Config(
                v_noise_stddev_percent_vmax=stddev, w_noise_stddev_percent_wmax=stddev
            ),
            WaypointController(ctl_cfg),
        )
        all_x = []
        u = np.full((2,), np.nan, dtype=np.float32)

        for wp_idx in range(len(waypoints)):
            traj, u_all = sim.navigate(x, u, waypoints[wp_idx])
            x = traj[-1]
            u = u_all[-1] if len(u_all) > 0 else u
            print(f"found waypoint {wp_idx} (it={len(traj)})")
            all_x.extend(traj)

        groundtruth_map = Map.construct_from_dataset(ds, cfg)
        plot_path_on_map(
            map,
            axes[idx],
            path,
            groundtruth_map,
            plot_centers=False,
            show_full_map=False,
        )
        plot_trajectory_over_waypoints(
            axes[idx], np.array(all_x), waypoints, sim.c.dist_thresh_m
        )
        total_t = round(sim.c.dt * len(all_x), 2)
        axes[idx].set_title(
            f"S={start_loc}, G={goal_loc} (#iter={len(all_x)}, t={total_t}s)"
        )

    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.suptitle(
        f"Q9: Online A*, post-hoc control (v, w noise stddev={stddev})",
        fontsize=16,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.15)
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
    fig = plt.figure("Q8", figsize=(20, 6))
    axes = fig.subplots(1, 4)

    for std, ax in zip(stddevs, axes):
        # target the waypoints above
        sim_cfg = RobotNavSim.Config(
            dt=dt,
            v_noise_stddev_percent_vmax=std,
            w_noise_stddev_percent_wmax=std,
            dist_thresh_m=0.5,
        )
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
        ax.set_title(f"u noise stddev={std * 100}% max (#iter={len(all_x)})")

    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.suptitle(
        f"Q8: Motion control for 5 waypoints, dt={dt}",
        fontsize=16,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.15)
    fig.show()


def q7(ds: Dataset):
    print("Part A, Question 7:")

    starts = [(2.45, -3.55), (4.95, -0.05), (-0.55, 1.45)]
    goals = [(0.95, -1.55), (2.45, 0.25), (1.95, 3.95)]

    fig = plt.figure("Q7", figsize=(10, 6))
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
        plot_path_on_map(
            map,
            axes[idx],
            path,
            groundtruth_map,
            plot_centers=False,
            show_full_map=False,
        )
        axes[idx].set_title(f"S={start_loc}, G={goal_loc}")

    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.suptitle(
        "Q7: Online A* paths (cell size = .1x.1m)", fontsize=16, fontweight="bold"
    )
    fig.subplots_adjust(bottom=0.15)
    fig.show()


def q5(ds: Dataset):
    print("Part A, Question 5:")

    starts = [(0.5, -1.5), (4.5, 3.5), (-0.5, 5.5)]
    goals = [(0.5, 1.5), (4.5, -1.5), (1.5, -3.5)]

    fig = plt.figure("Q5", figsize=(10, 6))
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
    fig.subplots_adjust(bottom=0.15)
    fig.show()


def q3(ds: Dataset):
    print("Part A, Question 3:")

    starts = [(0.5, -1.5), (4.5, 3.5), (-0.5, 5.5)]
    goals = [(0.5, 1.5), (4.5, -1.5), (1.5, -3.5)]

    fig = plt.figure("Q3", figsize=(10, 6))
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
