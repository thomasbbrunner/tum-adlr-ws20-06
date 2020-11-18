
from abc import ABC, abstractmethod
import importlib
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


class RobotSim3D(RobotSim):
    # create example of 3d with 3dof
    # (http://courses.csail.mit.edu/6.141/spring2011/pub/lectures/Lec14-Manipulation-II.pdf) slide 29
    pass


class RobotSim2D(RobotSim):
    """Simulation of 2D robotic arm with two or three revolute joints.

    Usage example for robot with three links:

    robot = RobotSim2D(3, [3, 2, 1])
    theta = [-np.pi/4, 0, 0]

    xtcp = robot.forward(theta)
    robot.plot_configurations(theta)

    theta_inv = robot.inverse(xtcp)

    """

    def __init__(self, num_links, len_links):

        self.num_links = num_links

        if type(len_links) is int or type(len_links) is float:
            self.len_links = np.array([len_links])
        else:
            self.len_links = np.array(len_links)

        if self.num_links not in [2, 3]:
            raise RuntimeError(
                "Unsupported number of links: {}".format(self.num_links))

        if len(self.len_links) != self.num_links:
            raise RuntimeError(
                "Missing link length information: {}". format(self.len_links))

        if not np.all(np.greater_equal(self.len_links, 0)):
            raise RuntimeError(
                "Link length has to be non-negative: {}". format(self.len_links))

        # resize to be able to use same formulas for 2 and 3 links
        self.len_links.resize(3)

    def forward(self, joint_states):
        """Returns TCP coordinates for specified joint states.
        Also accepts batch processing of several joint states.

        Examples:
        >>> # for single state
        >>> robot.forward(
                [0, 1, 0])

        >>> # for batch of two states
        >>> robot.forward([
                [0, 1, 0],
                [1, 0, 1]])
        """

        # make sure it is a numpy array
        joint_states = np.array(joint_states)

        input_dim = joint_states.ndim

        if input_dim == 1:
            joint_states = np.expand_dims(joint_states, axis=0)

        if joint_states.shape[1] != self.num_links:
            raise RuntimeError(
                "Missing joint state information: {}". format(joint_states))

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

        if input_dim == 1:
            return tcp_coordinates.flatten()

        return tcp_coordinates

    def inverse(self, tcp_coordinates: list):
        """Returns joint states for specified TCP coordinates.

        tcp_coordinates are the (x, y, phi) coordinates of the manipulator.
        """

        if len(tcp_coordinates) != 3:
            raise RuntimeError(
                "Missing TCP coordinates information: {}". format(tcp_coordinates))

        xtcp = tcp_coordinates[0]
        ytcp = tcp_coordinates[1]
        phi = tcp_coordinates[2]
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
        theta3_2 = phi - theta1_2 - theta2_1
        theta3_3 = phi - theta1_1 - theta2_2
        theta3_4 = phi - theta1_2 - theta2_2

        theta = np.array([
            [theta1_1, theta2_1, theta3_1],
            [theta1_2, theta2_1, theta3_2],
            [theta1_1, theta2_2, theta3_3],
            [theta1_2, theta2_2, theta3_4],
        ])

        return theta

    def plot_configurations(self, joint_states):
        """
        """

        if type(joint_states) is int or type(joint_states) is float:
            joint_states = np.array([[joint_states]])
        else:
            joint_states = np.array(joint_states)

            if joint_states.ndim == 1:
                joint_states = np.expand_dims(joint_states, axis=0)
            elif joint_states.ndim > 2:
                raise RuntimeError(
                    "Expected fewer dimensions for joint_states: {}". format(joint_states))

        joint_coords = np.zeros((joint_states.shape[0], 4, 4))

        for i, theta in enumerate(joint_states):

            if theta.shape[0] != self.num_links:
                raise RuntimeError(
                    "Missing joint state information: {}". format(joint_states))

            theta.resize(3)

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
                v1.flatten(), v2.flatten(),
                v3.flatten(), vtcp.flatten()]

        # import matplotlib only if needed
        plt = importlib.import_module(".pyplot", "matplotlib")

        fig, ax = plt.subplots()
        ax.grid()
        robot_length = np.sum(self.len_links)*1.1
        ax.set_xlim([-robot_length, robot_length])
        ax.set_ylim([-robot_length, robot_length])

        ax.plot(
            joint_coords[:, :, 0].flatten(),
            joint_coords[:, :, 1].flatten(),
            'or')

        ax.plot(
            joint_coords[:, :, 0].flatten(),
            joint_coords[:, :, 1].flatten(),)

        plt.show()
