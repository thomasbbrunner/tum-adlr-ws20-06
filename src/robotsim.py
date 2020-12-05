
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pdb


class RobotSim(ABC):

    @abstractmethod
    def forward(self, joint_states):
        """Returns TCP coordinates for specified joint states.

        TCP coordinates have format (x, y, phi)
            x,y:  coordinates of TCP
            phi:  angle of TCP with respect to
                  horizontal in range [-pi, pi).

        Also accepts batch processing of several joint states.

        Examples:
        >>> # for a single state and a robot with three joints
        >>> robot.forward(
                [0, 1, 0])

        >>> # for batch of two states and a robot with three joints
        >>> robot.forward([
                [0, 1, 0],
                [1, 0, 1]])
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def inverse_sampling(self, tcp_coordinates, step=None, num_samples=None):
        """Generates samples of inverse kinematics
        based on (x, y) TCP coordinates.

        You must provide either step or num_samples, but not both!

        Also accepts batch processing of several TCP coordinates.
        In this case, step and num_samples are applied to
        each set of TCP coordinates.

        Based on approximate Bayesian computation.

        "step" and "num_samples" are applied to all sampling variables.

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
        pass

    @abstractmethod
    def get_joint_coords(self, joint_states):
        pass

    @staticmethod
    def dh_transformation(alpha, a, d, theta, squeeze_dims=True):
        """Returns transformation matrix between two frames
        according to the Denavit-Hartenberg convention presented
        in 'Introduction to Robotics' by Craig.

        Also accepts batch processing of several joint states (d or theta).

        Transformation from frame i to frame i-1:
        alpha:  alpha_{i-1}
        a:      a_{i-1}
        d:      d_i     (variable in prismatic joints)
        theta:  theta_i (variable in revolute joints)
        """

        d = np.atleast_1d(d)
        theta = np.atleast_1d(theta)

        if d.shape[0] > 1 and theta.shape[0] > 1:
            raise RuntimeError(
                "Only one variable joint state is allowed.")

        desired_shape = np.maximum(d.shape[0], theta.shape[0])

        alpha = np.resize(alpha, desired_shape)
        a = np.resize(a, desired_shape)
        d = np.resize(d, desired_shape)
        theta = np.resize(theta, desired_shape)
        zeros = np.zeros(desired_shape)
        ones = np.ones(desired_shape)

        sin = np.sin
        cos = np.cos
        th = theta
        al = alpha

        transformation = np.array([
            [cos(th),           -sin(th),           zeros,          a],
            [sin(th)*cos(al),   cos(th) * cos(al),  -sin(al),   -sin(al)*d],
            [sin(th)*sin(al),   cos(th) * sin(al),  cos(al),    cos(al)*d],
            [zeros,             zeros,              zeros,      ones]
        ])

        # fix dimensions
        transformation = np.rollaxis(transformation, 2)

        if squeeze_dims:
            transformation = np.squeeze(transformation)

        return transformation

    def plot(self, joint_states, path=None, separate_plots=True, show=False):
        """Plots robot for specified joint states.

        Also accepts batches of joint states.

        If a path is provided, the plot is saved to an image.

        Plot is only shown if "show" is set.

        Examples:
        >>> # for single plot
        >>> robot.plot_configurations([0, 1, 0])

        >>> # for single plot with two configurations
        >>> robot.plot([
                [0, 1, 0],
                [1, 0, 1]],
                separate_plots=False)

        >>> # for 2 plots for each configuration
        >>> robot.plot([
                [0, 1, 0],
                [1, 0, 1]])
        """

        joint_states = np.atleast_2d(np.array(joint_states))

        # TODO fix here size of joint states
        _joint_states = np.zeros((joint_states.shape[0], 3))
        _joint_states[
            :joint_states.shape[0],
            :joint_states.shape[1]] = joint_states
        joint_states = _joint_states

        joint_coords = self.get_joint_coords(joint_states)

        if separate_plots:

            for arm in joint_coords:

                fig, ax = plt.subplots()
                ax.grid()
                # TODO rethink plot limits
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


class Robot3D1(RobotSim):
    # create example of 3d with 3dof
    # http://courses.csail.mit.edu/6.141/spring2011/pub/lectures/Lec14-Manipulation-II.pdf (slide 29)
    pass


class Robot2D2DoF(RobotSim):
    """Simulation of 2D robotic arm with two revolute joints.

    Examples:
    >>> robot = Robot2DoF([3, 2])
    """

    def __init__(self, len_links):
        self.sim = Robot2D3DoF([*len_links, 0])
        self.len_links = self.sim.len_links

    def forward(self, joint_states):

        joint_states = np.atleast_2d(joint_states)
        joint_states = np.hstack(
            (joint_states, np.zeros((joint_states.shape[0], 1))))

        return self.sim.forward(joint_states)

    def inverse(self, tcp_coordinates):

        joint_states = self.sim.inverse(tcp_coordinates)

        # TODO remove zero values for joint 3

        return joint_states

    def inverse_sampling(self, tcp_coordinates, step=None, num_samples=None):

        joint_states = self.sim.inverse_sampling(
            tcp_coordinates, step, num_samples)

        # TODO remove zero values for joint 3

        return joint_states

    def get_joint_coords(self, joint_states):

        joint_states = np.atleast_2d(joint_states)
        joint_states = np.hstack(
            (joint_states, np.zeros((joint_states.shape[0], 1))))

        return self.sim.get_joint_coords(joint_states)


class Robot2D3DoF(RobotSim):
    """Simulation of 2D robotic arm with three revolute joints.

    Examples:
    >>> robot = Robot3DoF([3, 2, 1])
    """

    NUM_LINKS = 3

    def __init__(self, len_links):

        self.len_links = np.array(len_links)

        if self.len_links.shape[0] != self.NUM_LINKS:
            raise RuntimeError(
                "Conflicting link lengths: {}". format(self.len_links))

        if not np.all(np.greater_equal(self.len_links, 0)):
            raise RuntimeError(
                "Link length has to be non-negative: {}". format(self.len_links))

        # resize to be able to use same formulas for 2 and 3 links
        self.len_links.resize(3)

    def forward(self, joint_states):

        joint_states = np.array(joint_states)
        input_dim = joint_states.ndim

        if input_dim == 1:
            joint_states = np.expand_dims(joint_states, axis=0)

        if joint_states.shape[1] != self.NUM_LINKS:
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

        if ((step is None and num_samples is None) or
                (step is not None and num_samples is not None)):
            raise RuntimeError(
                "Please provide either a step or a num_samples value.")

        if step is not None:
            tcp_angles = np.arange(-np.pi, np.pi, step)
        elif num_samples is not None:
            tcp_angles = np.linspace(-np.pi, np.pi, num_samples)

        tcp_coordinates = np.repeat(
            np.atleast_2d(tcp_coordinates),
            tcp_angles.shape[0],
            axis=0)

        tcp_angles = np.resize(tcp_angles, (tcp_coordinates.shape[0], 1))
        tcp_coordinates = np.hstack((tcp_coordinates, tcp_angles))

        joint_states = self.inverse(tcp_coordinates)

        # remove invalid solutions
        joint_states = joint_states[~np.isnan(joint_states)]
        joint_states = np.reshape(joint_states, (-1, 3))

        return joint_states

    def get_joint_coords(self, joint_states):

        joint_states = np.atleast_2d(joint_states)

        T01 = self.dh_transformation(
            0, 0, 0, joint_states[:, 0], False)
        T02 = T01 @ self.dh_transformation(
            0, self.len_links[0], 0, joint_states[:, 1], False)
        T03 = T02 @ self.dh_transformation(
            0, self.len_links[1], 0, joint_states[:, 2], False)
        Ttcp = T03 @ self.dh_transformation(
            0, self.len_links[2], 0, 0, False)

        # last column of T contains vector to each joint
        joint_coords = np.swapaxes(
            np.array(
                [T01[:, :2, -1], T02[:, :2, -1], T03[:, :2, -1], Ttcp[:, :2, -1]]),
            0, 1)

        return joint_coords


class Robot2D4DoF(RobotSim):
    """Simulation of 2D robotic arm with a prismatic and three revolute joints.

    Examples:
    >>> robot = Robot2D4DoF([3, 3, 3])
    """

    def __init__(self, len_links):

        self.sim = Robot2D3DoF(len_links)

        self.length = np.sum(len_links)

    def forward(self, joint_states):

        tcp_coordinates = self.sim.forward(joint_states[1:])

        tcp_coordinates[1] += joint_states[0]

        return tcp_coordinates

    def inverse(self):
        raise RuntimeError(
            "This function cannot be implemented.")

    def inverse_sampling(self, tcp_coordinates, step=None, num_samples=None):

        if ((step is None and num_samples is None) or
                (step is not None and num_samples is not None)):
            raise RuntimeError(
                "Please provide either a step or a num_samples value.")

        # TODO test if base height range makes sense
        if step is not None:
            tcp_angles = np.arange(-np.pi, np.pi, step)
            base_height = np.arange(-self.length, self.length, step)

        elif num_samples is not None:
            tcp_angles = np.linspace(-np.pi, np.pi, num_samples)
            base_height = np.linspace(-self.length, self.length, num_samples)

        tcp_angles = np.repeat(tcp_angles, base_height.shape[0])
        base_height = np.resize(base_height, tcp_angles.shape[0])
        sampling_vars = np.vstack((tcp_angles, base_height)).T

        # move robot arm to x-axis
        # simplifies finding solution
        tcp_coordinates = np.atleast_2d(tcp_coordinates)
        shift = tcp_coordinates[:, 1].copy()
        tcp_coordinates[:, 1] -= shift

        tcp_coordinates = np.repeat(
            tcp_coordinates,
            sampling_vars.shape[0],
            axis=0)
        shift = np.repeat(
            shift,
            sampling_vars.shape[0])
        sampling_vars = np.resize(
            sampling_vars,
            (tcp_coordinates.shape[0], 2))

        tcp_coordinates = np.vstack(
            (tcp_coordinates[:, 0], sampling_vars[:, 1], sampling_vars[:, 0])).T

        joint_states = self.sim.inverse(tcp_coordinates)
        joint_states = np.reshape(joint_states, (-1, 3))

        # add shift back into states
        heights = np.reshape(shift + sampling_vars[:, 1], (-1, 1))
        joint_states = np.hstack(
            (np.reshape(np.repeat(heights, joint_states.shape[0]/tcp_coordinates.shape[0]), (-1, 1)),
             joint_states))

        # remove invalid solutions
        joint_states = joint_states[~np.isnan(joint_states).any(axis=1)]

        return joint_states


if __name__ == "__main__":

    # functionality tests

    # Paper's robot

    # robot = Planar(3, [1, 1, 3])

    # robot.inverse_sampling([[1, 2], [3, 4]], num_samples=3)
    # print(robot.inverse_sampling([1, 2], num_samples=5))

    # robot.plot_configurations()

    # # joint_states = robot.inverse_sampling([3.5, 0], num_samples=100)
    # # robot.plot_configurations(joint_states, separate_plots=False)

    # plt.show()

    # if False:

    # 2 DoF
    robot = Robot2D2DoF([3, 2])
    print(robot.forward([0, 1]))
    print(robot.forward([[0, 1], [1, 0], [1, 1]]))
    print(robot.inverse([[1, 1, 1], [1, 1, 0]]))

    robot.plot([0, 1], separate_plots=False)
    robot.plot([1, 0], separate_plots=True)
    robot.plot([[0, 1], [1, 0]], separate_plots=False)
    robot.plot([[0, 1], [1, 0]], separate_plots=True)

    # 3 DoF
    robot = Robot2D3DoF([3, 3, 3])
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
    robot.plot([1, 2, 3], separate_plots=False)
    robot.plot([-3, -2, -1], separate_plots=True)
    robot.plot([[1, 2, 3], [-3, -2, -1]], separate_plots=False)
    robot.plot([[1, 2, 3], [-3, -2, -1]], separate_plots=True)

    joint_states = robot.inverse_sampling([7, 3], num_samples=20)
    robot.plot(joint_states, separate_plots=False)
    joint_states = robot.inverse_sampling([7, 3], num_samples=400)
    robot.plot(joint_states, separate_plots=False)
    joint_states = robot.inverse_sampling(
        [[7, 3], [-7, -3], [7, -3]], num_samples=100)
    robot.plot(joint_states, separate_plots=False)

    plt.show()
