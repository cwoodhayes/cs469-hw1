"""
Controller code for the turtle, ensuring that it hits waypoints
"""

from dataclasses import dataclass
import numpy as np

from turtlesim_msgs.msg import Pose


class TurtleWaypointReached(Exception):
    """
    Raised when a waypoint is reached in tick()
    """

    pass


class TurtleWaypointController:
    """
    Encapsulates logic for controlling the turtle's motion, such that
    it travels towards a desired waypoint
    """

    @dataclass
    class Config:
        # proportional gain for omega
        K_p: float = 10.0
        # angular velocity with 0 error
        p_0: float = 0

        # min. linear velocity = (filter freq (Hz) * tolerance (m) * 2) * velocity_fudge_factor
        velocity_fudge_factor: float = 0.5

    def __init__(self, freq: float, tolerance: float = 0.05) -> None:
        """
        :param freq: approximate frequency at which tick() is called
        :param tolerance: waypoint arrival tolerance
        """
        self._pose = np.full(5, np.nan)
        # robot state as published by turtlesim, in the form (x, y, theta, forward_velocity, angular_velocity)

        self._config = self.Config()
        self.tolerance = tolerance
        self.freq = freq

    def tick(self, waypoint: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Executes one pass of the control loop; outputs linear & angular velocity

        :param waypoint: [x, y] for the point we're aiming for
        :return: [linear & angular velocity, current distance to waypoint]
        :rtype: (np.ndarray(v_forward, theta_dot), float)
        """
        # a P controller for the

        # some facts:
        # - we need our max distance traveled per update to be <tolerance when near the target, to make sure that we don't speed over the target
        # - ideally, we should slow down when approaching the target, but we don't need to (we can just go slow the whole time)
        # - if we hold linear velocity constant per the first fact above, we can just do a P controller on the forward orientation

        ###################### Begin_Citation [3] ############################

        min_linear_velocity = (
            self.freq * self.tolerance * 2
        ) * self._config.velocity_fudge_factor
        # just use the minimum all the time for now, and do a P controller for the angle

        # calculate angular error
        p_to_w = waypoint - self._pose[:2]
        dist = np.linalg.norm(p_to_w)
        if dist <= self.tolerance:
            # we don't need to do any other calcs; let the caller handle this & call tick() again with a new waypoint
            # if it wants
            raise TurtleWaypointReached()

        # arctan outputs in the range [-pi/2, pi/2], but we need [-pi, pi]. hence arctan2
        waypoint_heading = np.arctan2(p_to_w[1], p_to_w[0])
        err = self.angle_diff(waypoint_heading, self._pose[2])

        # units of err are radians; units of p_out should be rads/s. So we consider K_p as Hz
        p_out = self._config.K_p * err + self._config.p_0

        ###################### End_Citation [3] ############################
        return np.array((min_linear_velocity, p_out)), dist

    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        """
        Returns the shortest distance between 2 angles on the xy plane,
        independent of representation (i.e. the distance between -pi and 3pi is 0, not 4pi)

        :param a: angle 1
        :param b: angle 2
        :return: shortest angle between them
        """
        ########### Begin_Citation[4] ###################
        return np.arctan2(np.sin(a - b), np.cos(a - b))
        ########### End_Citation ########################

    def set_pose(self, pose: Pose) -> None:
        """
        Sets robot pose. No side-effects, just a setter
        """
        self._pose = np.array(
            (
                pose.x,
                pose.y,
                pose.theta,
                pose.linear_velocity,
                pose.angular_velocity,
            )
        )

    def get_pose(self) -> np.ndarray:
        """
        Get robot pose

        :param self: Description
        :return: Description
        :rtype: ndarray
        """
        return self._pose
