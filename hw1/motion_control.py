"""
Controller code for the robot, ensuring that it hits waypoints
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from hw1.motion_model import MotionModel


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
        self.c = self.Config() if config is None else config

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
        w_out = self.c.wK_p * theta_err + self.c.wp_0

        # control linear velocity:
        dist_err = np.linalg.norm(p_to_w)
        v_out = self.c.vK_p * dist_err + self.c.vp_0

        # enforce acceleration limits:
        vdot = v_out - u_prev[0]
        wdot = w_out - u_prev[1]
        if abs(vdot) > self.c.vdot_max:
            v_out = u_prev[0] + (self.c.vdot_max if vdot > 0 else -self.c.vdot_max)
        if abs(wdot) > self.c.wdot_max:
            w_out = u_prev[1] + (self.c.wdot_max if wdot > 0 else -self.c.wdot_max)

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


class RobotNavSim:
    """
    Simulates the robot as it navigates between waypoints
    """

    @dataclass
    class Config:
        max_iter: int = 100
        dist_thresh_m: float = 0.05
        dt: float = 0.1
        x_noise_stddev: float = 0.02

        w_noise_stddev_percent_wmax: float = 0.00
        v_noise_stddev_percent_vmax: float = 0.00

    def __init__(
        self,
        cfg: Config,
        controller: WaypointController,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.c = cfg
        self.ctl = controller
        self.motion = MotionModel()
        # default to seed=0 for repeatability
        self.rng = rng if rng is not None else np.random.default_rng(seed=0)

    @staticmethod
    def _point_is_out_of_bounds(point: np.ndarray, bounds: np.ndarray) -> bool:
        """Check if a point is out of bounds of a square region

        Args:
            coord (np.ndarray): (x,y)
            bounds (np.ndarray): 2x2 array ((xmin, xmax), (ymin, ymax)) of square region

        Returns:
            bool: true if OOB, false otherwise
        """
        if point[0] < bounds[0, 0] or point[0] > bounds[0, 1]:
            # check xlim
            return True
        if point[1] < bounds[1, 0] or point[1] > bounds[1, 1]:
            # check ylim
            return True
        return False

    def navigate(
        self,
        x0: np.ndarray,
        u_prev: np.ndarray,
        waypoint: np.ndarray,
        grid_bounds: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Navigate from x0 to waypoint and return the trajectory.

        Uses the motion model to propagate control output.
        Introduces gaussian noise to the motion model state output to
        simulate real robot control+measurement uncertainty.

        Args:
            x0 (np.ndarray): start state (x, y, theta)
            u_prev (np.ndarray): previous control output (v, w)
            waypoint (np.ndarray): destination location
            grid_bounds(ndarray): if given, a square region in (x,y). If the trajectory leaves this region, navigate()
            returns early. Form: ((xlim), (ylim))

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]:
                0: trajectory [(x, y, theta), ...] beginning with x0
                1: controls [(v, w), ...], not including u_prev
        """

        x = x0.copy()
        u = u_prev
        all_x = []
        all_u = []
        it = 0

        while it < self.c.max_iter:
            all_x.append(x)
            if np.linalg.norm(x[0:2] - waypoint) < self.c.dist_thresh_m:
                return all_x, all_u
            if grid_bounds is not None and self._point_is_out_of_bounds(
                x[0:2], grid_bounds
            ):
                return all_x, all_u
            u = self.ctl.tick(x, u, waypoint)
            u_noise = np.array(
                [
                    self.rng.normal(
                        0, self.ctl.c.vdot_max * self.c.v_noise_stddev_percent_vmax
                    ),
                    self.rng.normal(
                        0, self.ctl.c.wdot_max * self.c.w_noise_stddev_percent_wmax
                    ),
                ]
            )
            x_ideal = self.motion.tick(
                u + u_noise,
                x,
                self.c.dt,
            )
            x = x_ideal + self.rng.normal(0, self.c.x_noise_stddev, size=x.shape)
            all_u.append(u)
            it += 1

        raise RuntimeError(f"Unable to find waypoint in {it} iterations.")
