import pathlib
import signal

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from hw0.astar import AStar
from hw0.data import Dataset
from hw0.map import Map
from hw0.plot import plot_path_on_map
from hw0.online_astar import run_astar_online


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
    q5(ds)


def q5(ds: Dataset):
    print("Part A, Question 5:")

    starts = [(0.5, -1.5), (4.5, 3.5), (-0.5, 5.5)]
    goals = [(0.5, 1.5), (4.5, -1.5), (1.5, -3.5)]

    fig = plt.figure()
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

        path, map = run_astar_online(ds, cfg)
        path.print()

        plot_path_on_map(map, axes[idx], path)
        axes[idx].set_title(f"S={start_loc}, G={goal_loc}")

    plt.show()


def q3(ds: Dataset):
    print("Part A, Question 3:")

    starts = [(0.5, -1.5), (4.5, 3.5), (-0.5, 5.5)]
    goals = [(0.5, 1.5), (4.5, -1.5), (1.5, -3.5)]

    fig = plt.figure()
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

    plt.show()


if __name__ == "__main__":
    main()
