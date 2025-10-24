import numpy as np


class MotionModel:
    """
    Motion model for a diff-drive robot on a 2d plane.

    Adapted from Probabilistic Robotics, section 5.3, page 101
    """

    def tick(self, u: np.ndarray, x_prev: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagate state x forward dt in time

        :param u: Control input as a vector
        :type u: [forward velocity (m/s), angular velocity (rad/s)]
        :param x_prev: x_t-1 [x, y, theta]
        :return: new state vector
        :rtype: [x, y, theta]
        """
        x = np.zeros_like(x_prev)

        v = u[0]
        w = u[1]

        # theta_t = theta_t-1 + wdt
        x[2] = x_prev[2] + (w * dt)

        if (w * dt) == 0:
            # in this case, we have a straight line
            # (infinite radius, infinitesmal theta). handle that directly
            x[0] = x_prev[0] + (v * np.cos(x[2]))
            x[1] = x_prev[1] + (v * np.sin(x[2]))
            x[2] = x_prev[2]
        else:
            # radius r of the arc trajectory := |v|/wdt
            r = v / (w * dt)

            x_c = x_prev[0] - r * np.sin(x_prev[2])
            y_c = x_prev[1] + r * np.cos(x_prev[2])

            x[0] = x_c + r * (np.sin(x[2]))
            x[1] = y_c - r * (np.cos(x[2]))

        return x
