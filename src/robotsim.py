
from abc import ABC, abstractmethod
import importlib
import numpy as np
import pdb


def dh_transformation(alpha, a, d, theta):
    """Returns transformation matrix between two frames
    according to the Denavit-Hartenberg convention presented
    in [Craig: Introduction to robotics].

    T:      transformation from frame i to frame i-1
    alpha:  alpha_{i-1}
    a:      a_{i-1}
    d:      d_i
    theta:  theta_i
    """

    sin = np.sin
    cos = np.cos
    th = theta
    ap = alpha

    return np.array([
        [cos(th),           -sin(th),           0,          a],
        [sin(th)*cos(ap),   cos(th) * cos(ap),  -sin(ap),   -sin(ap)*d],
        [sin(th)*sin(ap),   cos(th) * sin(ap),  cos(ap),    cos(ap)*d],
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
    """Simulation of 2D robotic arm with up tp three revolute joints.

    Usage example for robot with three links:

    robot = RobotSim2D(3, [3, 2, 1])
    theta = [-np.pi/4, 0, 0]

    xtcp = robot.forward(theta)
    robot.plot_configurations(theta)

    theta_inv = robot.inverse(xtcp)

    """

    def __init__(self, num_links: int, len_links: list or int):

        if num_links not in [1, 2, 3]:
            raise RuntimeError(
                "Unsupported number of links: {}".format(num_links))

        if type(len_links) is int or type(len_links) is float:
            len_links = np.array([len_links])
        else:
            len_links = np.array(len_links)

        if len(len_links) != num_links:
            raise RuntimeError(
                "Missing link length information: {}". format(len_links))

        if not np.all(np.greater_equal(len_links, 0)):
            raise RuntimeError(
                "Link length has to be non-negative: {}". format(len_links))

        self.num_links = num_links
        self.len_links = np.zeros(3)
        self.len_links[:len_links.shape[0]] = len_links

    def forward(self, joint_states: list or int):
        """Returns TCP coordinates for specified joint states.
        """

        if type(joint_states) is int or type(joint_states) is float:
            joint_states = np.array([joint_states])
        else:
            joint_states = np.array(joint_states)

        if len(joint_states) != self.num_links:
            raise RuntimeError(
                "Missing joint state information: {}". format(joint_states))

        theta = np.zeros(3)
        theta[:joint_states.shape[0]] = joint_states

        xtcp = (
            self.len_links[0]*np.cos(theta[0]) +
            self.len_links[1]*np.cos(theta[0] + theta[1]) +
            self.len_links[2]*np.cos(theta[0] + theta[1] + theta[2]))
        ytcp = (
            self.len_links[0]*np.sin(theta[0]) +
            self.len_links[1]*np.sin(theta[0] + theta[1]) +
            self.len_links[2]*np.sin(theta[0] + theta[1] + theta[2]))
        phi = theta[0] + theta[1] + theta[2]

        return xtcp, ytcp, phi

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

    def plot_configurations(self, joint_states: list or int):
        """
        """

        # import matplotlib only if needed
        plt = importlib.import_module(".pyplot", "matplotlib")

        T01 = dh_transformation(0, 0, 0, joint_states[0])
        T02 = T01 @ dh_transformation(0, self.len_links[0], 0, joint_states[1])
        T03 = T02 @ dh_transformation(0, self.len_links[1], 0, joint_states[2])

        v1 = np.zeros((4, 1))
        v2 = v1 + T01 @ np.array([[self.len_links[0]], [0], [0], [0]])
        v3 = v2 + T02 @ np.array([[self.len_links[1]], [0], [0], [0]])
        vtcp = v3 + T03 @ np.array([[self.len_links[2]], [0], [0], [0]])

        fig, ax = plt.subplots()
        ax.grid()
        robot_length = np.sum(self.len_links)*1.1
        ax.set_xlim([-robot_length, robot_length])
        ax.set_ylim([-robot_length, robot_length])

        ax.plot(
            [v1[0], v2[0], v3[0], vtcp[0]],
            [v1[1], v2[1], v3[1], vtcp[1]],
            'or')

        ax.plot(
            [v1[0], v2[0], v3[0], vtcp[0]],
            [v1[1], v2[1], v3[1], vtcp[1]])

        plt.show()
