
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pdb


def dh_transformation(alpha, a, d, theta):
    """Returns transformation matrix between two frames
    according to the Denavit-Hartenberg convention presented
    in 'Introduction to Robotics' by Craig.

    Transformation from frame i to frame i-1:
    alpha:  alpha_{i-1}
    a:      a_{i-1}
    d:      d_i
    theta:  theta_i
    """

    sin = np.sin
    cos = np.cos
    th = theta
    al = alpha

    return np.array([
        [cos(th),           -sin(th),           0,          a],
        [sin(th)*cos(al),   cos(th) * cos(al),  -sin(al),   -sin(al)*d],
        [sin(th)*sin(al),   cos(th) * sin(al),  cos(al),    cos(al)*d],
        [0,                 0,                  0,          1]
    ])


class RobotSim(ABC):

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def inverse(self):
        pass

    @abstractmethod
    def plot_configurations(self):
        pass


class RobotSim3D(RobotSim):
    # create example of 3d with 3dof
    # http://courses.csail.mit.edu/6.141/spring2011/pub/lectures/Lec14-Manipulation-II.pdf (slide 29)
    pass


class RobotSim2D(RobotSim):
    """Simulation of 2D robotic arm with two or three revolute joints.

    Examples:
    >>> # for robot with three links
    >>> robot = RobotSim2D(3, [3, 2, 1])

    >>> # for robot with two links
    >>> robot = RobotSim2D(2, [0.5, 0.5])
    """

    def __init__(self, num_links, len_links):

        self.num_links = num_links
        self.len_links = np.array(len_links)

        if self.num_links not in [2, 3]:
            raise RuntimeError(
                "Unsupported number of links: {}".format(self.num_links))

        if self.len_links.shape[0] != self.num_links:
            raise RuntimeError(
                "Conflicting link lengths: {}". format(self.len_links))

        if not np.all(np.greater_equal(self.len_links, 0)):
            raise RuntimeError(
                "Link length has to be non-negative: {}". format(self.len_links))

        # resize to be able to use same formulas for 2 and 3 links
        self.len_links.resize(3)

    def forward(self, joint_states):
        """Returns TCP coordinates for specified joint states.

        TCP coordinates have format (x, y, phi)
            x,y:  coordinates of TCP
            phi:  angle of TCP with respect to
                  horizontal in range [-pi, pi).

        Also accepts batch processing of several joint states.

        Examples:
        >>> # for a single state
        >>> robot.forward(
                [0, 1, 0])

        >>> # for batch of two states
        >>> robot.forward([
                [0, 1, 0],
                [1, 0, 1]])
        """

        joint_states = np.array(joint_states)
        input_dim = joint_states.ndim

        if input_dim == 1:
            joint_states = np.expand_dims(joint_states, axis=0)

        if joint_states.shape[1] != self.num_links:
            raise RuntimeError(
                "Conflicting joint states: {}". format(joint_states))

        theta = np.zeros((joint_states.shape[0], 3))
        theta[:, :joint_states.shape[1]] = joint_states

        tcp_coordinates = np.array([
            # x-coordinate of TCP
            self.len_links[0]*np.cos(theta[:, 0]) +
            self.len_links[1]*np.cos(theta[:, 0] + theta[:, 1]) +
            self.len_links[2]*np.cos(theta[:, 0] + theta[:, 1] + theta[:, 2]),
            # y-coordinate of TCP
            self.len_links[0]*np.sin(theta[:, 0]) +
            self.len_links[1]*np.sin(theta[:, 0] + theta[:, 1]) +
            self.len_links[2]*np.sin(theta[:, 0] + theta[:, 1] + theta[:, 2]),
            # angle of TCP with respect to horizontal
            theta[:, 0] + theta[:, 1] + theta[:, 2]])

        # limit angle of TCP to range [0, 2*pi)
        # tcp_coordinates[2] = tcp_coordinates[2] % (2*np.pi)

        # limit angle of TCP to range [-pi, pi)
        tcp_coordinates[2] = (tcp_coordinates[2] + np.pi) % (2*np.pi) - np.pi

        if input_dim == 1:
            return tcp_coordinates.flatten()

        return tcp_coordinates.T

    def inverse(self, tcp_coordinates):
        """Returns joint states for specified TCP coordinates.

        Inputs are the (x, y, phi) coordinates of the manipulator.

        Also accepts batch processing of several TCP coordinates.

        Returns at most two possible solutions for each input.
        If no solution is found, np.nan is returned.

        Examples:
        >>> # for a single coordinate
        >>> robot.inverse([3, -4, 0])

        >>> # for batch of two coordinates
        >>> robot.inverse([
                [3, -4, 0],
                [5, 1, 0.3]])
        """

        tcp_coordinates = np.array(tcp_coordinates)
        input_dim = tcp_coordinates.ndim

        if input_dim == 1:
            tcp_coordinates = np.expand_dims(tcp_coordinates, axis=0)

        if tcp_coordinates.shape[1] != 3:
            raise RuntimeError(
                "Missing TCP coordinates information: {}". format(tcp_coordinates))

        xtcp = tcp_coordinates[:, 0]
        ytcp = tcp_coordinates[:, 1]
        phi = tcp_coordinates[:, 2]
        l1 = self.len_links[0]
        l2 = self.len_links[1]
        l3 = self.len_links[2]

        x2 = xtcp - l3*np.cos(phi)
        y2 = ytcp - l3*np.sin(phi)

        c2 = (np.power(x2, 2) + np.power(y2, 2) - l1**2 - l2**2) / (2*l1*l2)

        s2_1 = np.sqrt(1 - np.power(c2, 2))
        s2_2 = -s2_1

        theta1_1 = np.arctan2(y2, x2) - np.arctan2(l2*s2_1, l1 + l2*c2)
        theta1_2 = np.arctan2(y2, x2) - np.arctan2(l2*s2_2, l1 + l2*c2)

        theta2_1 = np.arctan2(s2_1, c2)
        theta2_2 = np.arctan2(s2_2, c2)

        theta3_1 = phi - theta1_1 - theta2_1
        theta3_2 = phi - theta1_2 - theta2_2

        theta = np.array([
            [theta1_1, theta1_2],
            [theta2_1, theta2_2],
            [theta3_1, theta3_2],
        ])

        if input_dim == 1:
            return theta.T.reshape((2, 3))

        return theta.T.reshape((tcp_coordinates.shape[0], 2, 3))

    def inverse_sampling(self, tcp_coordinates, step=None, num_samples=None):
        """Generates samples of inverse kinematics 
        based on (x, y) TCP coordinates.

        You must provide either step or num_samples, but not both!

        Also accepts batch processing of several TCP coordinates.
        In this case, step and num_samples are applied to 
        each set of TCP coordinates.

        Based on approximate Bayesian computation.

        Args:
            tcp_coordinates: (x, y) coordinates of TCP.
            step: specifies the step size between samples.
            num_samples: specifies number of samples to generate.

        Examples:
        >>> # for a single coordinate and step
        >>> robot.full_inverse([7, 3], step=0.1)

        >>> # for a single coordinate and number of samples
        >>> robot.full_inverse([7, 3], num_samples=20)

        >>> # for batch of three coordinates and total of 300 samples
        >>> robot.full_inverse(
                [[7, 3], [-7, -3], [7, -3]], num_samples=100)
        """

        if ((step is None and num_samples is None) or
                (step is not None and num_samples is not None)):
            raise RuntimeError(
                "Please provide either a step or a num_samples value.")

        if step is not None:
            angles = np.arange(-np.pi, np.pi, step)
        elif num_samples is not None:
            angles = np.linspace(-np.pi, np.pi, num_samples)

        tcp_coordinates = np.repeat(
            np.atleast_2d(tcp_coordinates),
            angles.shape[0],
            axis=0)

        angles = np.resize(angles, (tcp_coordinates.shape[0], 1))
        tcp_coordinates = np.hstack((tcp_coordinates, angles))

        joint_states = self.inverse(tcp_coordinates)

        # remove invalid solutions
        joint_states = joint_states[~np.isnan(joint_states)]

        # stack solutions into 2D array
        joint_states = np.reshape(joint_states, (-1, 3))

        return joint_states

    def plot_configurations(self, joint_states, path=None, separate_plots=True, show=False):
        """Plots configurations for specified joint states.

        Also accepts batches of joint states.

        If a path is provided, the plot is saved to an image.

        Plot is only shown if "show" is set.

        TODO:
            - improve appearance
            - fix bugs

        Examples:
        >>> # for single plot
        >>> robot.plot_configurations([0, 1, 0])

        >>> # for single plot with two configurations
        >>> robot.plot_configurations([
                [0, 1, 0],
                [1, 0, 1]],
                separate_plots=False)

        >>> # for plot with two subplots
        >>> robot.plot_configurations([
                [0, 1, 0],
                [1, 0, 1]])
        """

        joint_states = np.atleast_2d(np.array(joint_states))

        _joint_states = np.zeros((joint_states.shape[0], 3))
        _joint_states[
            :joint_states.shape[0],
            :joint_states.shape[1]] = joint_states
        joint_states = _joint_states

        joint_coords = np.zeros((joint_states.shape[0], 4, 2))

        for i, theta in enumerate(joint_states):

            T01 = dh_transformation(0, 0, 0, theta[0])
            T02 = T01 @ dh_transformation(
                0, self.len_links[0], 0, theta[1])
            T03 = T02 @ dh_transformation(
                0, self.len_links[1], 0, theta[2])

            v1 = np.zeros((4, 1))
            v2 = v1 + T01 @ np.array([[self.len_links[0]], [0], [0], [0]])
            v3 = v2 + T02 @ np.array([[self.len_links[1]], [0], [0], [0]])
            vtcp = v3 + T03 @ np.array([[self.len_links[2]], [0], [0], [0]])

            joint_coords[i] = [
                v1[0:2].flatten(), v2[0:2].flatten(),
                v3[0:2].flatten(), vtcp[0:2].flatten()]

        if separate_plots:

            for arm in joint_coords:

                fig, ax = plt.subplots()
                ax.grid()
                robot_length = np.sum(self.len_links)*1.1
                ax.set_xlim([-robot_length, robot_length])
                ax.set_ylim([-robot_length, robot_length])

                ax.plot(
                    arm[:, 0].flatten(),
                    arm[:, 1].flatten(),
                    c='b')

                ax.scatter(
                    arm[:, 0].flatten(),
                    arm[:, 1].flatten(),
                    c='r', s=8)
        else:

            fig, ax = plt.subplots()
            ax.grid()
            robot_length = np.sum(self.len_links)*1.1
            ax.set_xlim([-robot_length, robot_length])
            ax.set_ylim([-robot_length, robot_length])

            # attempt at automatic transparency
            alpha = np.clip(
                np.exp(-0.008*joint_coords.shape[0]),
                0.01, 1)

            for arm in joint_coords:
                ax.plot(
                    arm[:, 0].flatten(),
                    arm[:, 1].flatten(),
                    c='b', alpha=alpha)

            ax.scatter(
                joint_coords[:, :, 0].flatten(),
                joint_coords[:, :, 1].flatten(),
                c='r', s=8)

        if path:
            plt.savefig(path)

        if show:
            plt.show()


if __name__ == "__main__":

    # functionality tests

    # 2 DoF
    robot = RobotSim2D(2, [3, 2])
    print(robot.forward([0, 1]))
    print(robot.forward([[0, 1], [1, 0], [1, 1]]))
    print(robot.inverse([[1, 1, 1], [1, 1, 0]]))

    robot.plot_configurations([0, 1], separate_plots=False)
    robot.plot_configurations([1, 0], separate_plots=True)
    robot.plot_configurations([[0, 1], [1, 0]], separate_plots=False)
    robot.plot_configurations([[0, 1], [1, 0]], separate_plots=True)

    # 3 DoF
    robot = RobotSim2D(3, [3, 3, 3])
    print(robot.forward([0, 1, 1]))
    print(robot.forward([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
    print(robot.inverse([3, -4, 0]))
    # # [[-2.411865    1.68213734  0.72972766]
    # #  [-0.72972766 -1.68213734  2.411865  ]]
    print(robot.inverse([[2, 2, 2], [5, 1, 0.3], [50, 1, 0.3]]))
    # # [[[-1.2030684   1.9652703   1.2377981 ]
    # #   [ 0.7622019  -1.9652703   3.2030684 ]]
    # #  [[-1.15352504  2.41326677 -0.95974173]
    # #   [ 1.25974173 -2.41326677  1.45352504]]
    # #  [[        nan         nan         nan]
    # #   [        nan         nan         nan]]]
    robot.plot_configurations([1, 2, 3], separate_plots=False)
    robot.plot_configurations([-3, -2, -1], separate_plots=True)
    robot.plot_configurations([[1, 2, 3], [-3, -2, -1]], separate_plots=False)
    robot.plot_configurations([[1, 2, 3], [-3, -2, -1]], separate_plots=True)

    joint_states = robot.inverse_sampling([7, 3], num_samples=20)
    robot.plot_configurations(joint_states, separate_plots=False)
    joint_states = robot.inverse_sampling([7, 3], num_samples=400)
    robot.plot_configurations(joint_states, separate_plots=False)
    joint_states = robot.inverse_sampling(
        [[7, 3], [-7, -3], [7, -3]], num_samples=100)
    robot.plot_configurations(joint_states, separate_plots=False)

    plt.show()
