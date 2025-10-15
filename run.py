import pathlib
import signal

from matplotlib import pyplot as plt
import numpy as np

from hw0.data import Dataset
from hw0.map import Map
from hw0.plot import plot_map


REPO_ROOT = pathlib.Path(__file__).parent


def main():
    print("cs469 Homework 1")
    # make matplotlib responsive to ctrl+c
    # cite: this stackoverflow answer:
    # https://stackoverflow.com/questions/67977761/how-to-make-plt-show-responsive-to-ctrl-c
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # my assigned dataset is ds1, so I'm hardcoding this
    ds = Dataset.from_dataset_directory(REPO_ROOT / "data/ds1")

    q1(ds)


def q1(ds: Dataset):
    print("Question 1:")

    cfg = Map.Config(
        dimensions=np.array(
            [
                [-2, 5],
                [-6, 6],
            ]
        ),
        cell_size=1.0,
        start=np.array([0.5, -1.5]),
        goal=np.array([0.5, 1.5]),
    )

    map = Map.construct_from_dataset(ds, cfg)

    fig = plt.figure()
    ax = fig.subplots()

    plot_map(map, ax)
    plt.show()


if __name__ == "__main__":
    main()
