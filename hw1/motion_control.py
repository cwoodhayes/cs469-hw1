"""
Controller code for the robot, ensuring that it hits waypoints
"""

from dataclasses import dataclass
import numpy as np


class WaypointController:
    """
    PID controller for a diff-drive robot.

    Given current position feedback, output angular & linear velocity control
    """

    @dataclass
    class Config:
        # proportional gain for v (forward speed)
        vK_p: float = 0.2
        # v bias - min. speed (with 0 error)
        vp_0: float = 0.15

        # proportional gain for omega
        wK_p: float = 10.0
        # omega bias - min. angular velocity (with 0 error)
        wp_0: float = 0.0

        # max accel (m/s^2)
        vdot_max: float = 0.288
        # max angular accel (rad/s^2)
        wdot_max: float = 5.579

    def __init__(self, config: Config | None = None) -> None:
        """
        :param config: controller configuration
        """
        self._c = self.Config() if config is None else config

    def tick(
        self, x_curr: np.ndarray, u_prev: np.ndarray, waypoint: np.ndarray
    ) -> np.ndarray:
        """
        Executes one pass of the control loop; outputs linear & angular velocity

        :param x_curr: robot state in the form [x, y, theta]
        :param u_prev: control output from the previous timestep [v, w]
        :param waypoint: [x, y] for the point we're aiming for
        :return: [v,w] control output
        """
        # 2 separate P controllers, one for forward velocity, the other for angular velocity

        p_to_w = waypoint - x_curr[0:2]

        # control angular velocity:
        # waypoint heading in range [-pi, pi]
        waypoint_heading = np.arctan2(p_to_w[1], p_to_w[0])
        theta_err = self.angle_diff(waypoint_heading, x_curr[2])

        # units of err are radians; units of p_out should be rads/s. So we consider K_p as Hz
        w_out = self._c.wK_p * theta_err + self._c.wp_0

        # control linear velocity:
        dist_err = np.linalg.norm(p_to_w)
        v_out = self._c.vK_p * dist_err + self._c.vp_0

        # enforce acceleration limits:
        vdot = v_out - u_prev[0]
        wdot = w_out - u_prev[1]
        if abs(vdot) > self._c.vdot_max:
            v_out = u_prev[0] + (self._c.vdot_max if vdot > 0 else -self._c.vdot_max)
        if abs(wdot) > self._c.wdot_max:
            w_out = u_prev[1] + (self._c.wdot_max if wdot > 0 else -self._c.wdot_max)

        return np.array((v_out, w_out))

    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        """
        Returns the shortest distance between 2 angles on the xy plane,
        independent of representation (i.e. the distance between -pi and 3pi is 0, not 4pi)

        :param a: angle 1
        :param b: angle 2
        :return: shortest angle between them
        """
        return np.arctan2(np.sin(a - b), np.cos(a - b))
