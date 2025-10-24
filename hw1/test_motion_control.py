"""Tests for motion_controller.py"""

import numpy as np

from hw1 import motion_control


def test_waypoint_controller_basic() -> None:
    # use default config
    ctl = motion_control.WaypointController()

    x_curr = np.array([0.0, 0.0, 0.0])
    u_prev = np.array([0.0, 0.0])
    wp1 = np.array([100.0, 100.0])

    # run to a waypoint that's really far away in both theta and dist,
    # and make sure the accel gets correctly latched.

    u = ctl.tick(x_curr, u_prev, wp1)

    assert u[0] == 0.288
    assert u[1] == 5.579

    # run to a waypoint that's our current location, and make sure
    # our biases are obeyed
    u = ctl.tick(x_curr, u_prev, np.array([0.0, 0.0]))
    assert u[0] == 0.15
    assert u[1] == 0.0
