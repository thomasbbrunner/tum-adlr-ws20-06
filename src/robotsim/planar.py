
from abc import ABC, abstractmethod
import numpy as np

from robotsim.plot import plot, heatmap
from robotsim.robot import Robot
import utils


class Planar(Robot):
    """Simulation of planar robot with any number of revolute joints.
    For documentation, please refer to the base class.
    """

    def __init__(self, num_dof, len_links):

        self._num_dof = num_dof
        self._len_links = np.array(len_links)
        self._length = np.sum(self._len_links)

        if not isinstance(self._num_dof, int):
            raise RuntimeError(
                "Number of degrees of freedoms has to be an integer: {}"
                .format(self._num_dof))

        if self._len_links.shape != (self._num_dof,):
            raise RuntimeError(
                "Wrong shape of link lengths: {}"
                .format(self._len_links))

        if not np.all(np.greater_equal(self._len_links, 0)):
            raise RuntimeError(
                "Link length has to be non-negative: {}"
                .format(self._len_links))

        # pre-computed arrays for calculations
        self._angle_selector = np.triu(
            np.ones((self._num_dof, self._num_dof)))

    def forward(self, joint_states, squeeze=True):

        joint_states = np.atleast_2d(joint_states)

        if joint_states.shape[1] != self._num_dof:
            raise RuntimeError(
                "Wrong shape of joint states: {}"
                .format(joint_states))

        # generate matrix with sums of angles
        # [theta1, theta1+theta2, theta1+theta2+theta3, ...]
        thetas = joint_states @ self._angle_selector

        x_tcp = np.sum(np.cos(thetas) * self._len_links, axis=1)
        y_tcp = np.sum(np.sin(thetas) * self._len_links, axis=1)
        phi_tcp = utils.wrap(thetas[:, -1])
        tcp = np.vstack((x_tcp, y_tcp, phi_tcp)).T

        if squeeze:
            tcp = np.squeeze(tcp)

        return tcp

    def inverse(self, tcp_coordinates, squeeze=True):

        if self._num_dof == 3 or self._num_dof == 2:

            tcp_coordinates = np.atleast_2d(tcp_coordinates)
            if tcp_coordinates.shape[1] != 3:
                raise RuntimeError(
                    "Wrong shape of TCP coordinates: {}"
                    .format(tcp_coordinates))

            xtcp = tcp_coordinates[:, 0]
            ytcp = tcp_coordinates[:, 1]
            phi = tcp_coordinates[:, 2]
            len_links = np.zeros(3)
            len_links[:self._len_links.shape[0]] = self._len_links
            l1 = len_links[0]
            l2 = len_links[1]
            l3 = len_links[2]

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

            theta = utils.wrap(theta)
            theta = np.swapaxes(theta, 0, 2)

            if self._num_dof == 2:
                # remove solutions for joint 3
                theta = theta[:, :, :2]

            if squeeze:
                theta = np.squeeze(theta)

            return theta

        else:
            raise RuntimeError(
                "Function not implemented.")

    def get_joint_coords(self, joint_states):

        joint_states = np.atleast_2d(joint_states)

        if joint_states.shape[1] != self._num_dof:
            raise RuntimeError(
                "Wrong shape of joint states: {}"
                .format(joint_states))

        # dimensions: (num_inputs, joints, coords)
        joint_coords = np.zeros(
            (joint_states.shape[0], self._num_dof+1, 2))

        # first transformation has no link
        T = utils.dh_transformation(
            0, 0, 0, joint_states[:, 0], False)
        joint_coords[:, 0] = T[:, :2, -1]

        # coordinates of each link
        for i in range(1, self._num_dof):
            T = T @ utils.dh_transformation(
                0, self._len_links[i-1], 0, joint_states[:, i], False)
            joint_coords[:, i] = T[:, :2, -1]

        # coordinates of end-effector
        T = T @ utils.dh_transformation(
            0, self._len_links[-1], 0, 0, False)
        joint_coords[:, -1] = T[:, :2, -1]

        return joint_coords

    def get_joint_ranges(self):
        ranges = np.array([[-np.pi, np.pi]])
        ranges = np.repeat(ranges, self._num_dof, axis=0)
        return ranges

    def get_length(self):
        return self._length

    def get_dof(self):
        return self._num_dof


if __name__ == "__main__":

    # functionality tests
    robot = Planar(5, [1, 1, 1, 1, 2])

    # one input
    js = [1, -1, -1, 1, -1]
    # plot(js, robot, show=True)
    tcp = robot.forward(js)
    js_sampling = robot.rejection_sampling(
        tcp[:2], num_samples=5000, eps=0.05, mean=0, std=1)
    robot.get_joint_coords(js)
    heatmap(
        js_sampling, robot, highlight=1, transparency=0.005,
        path="./figures/robotsim/rejection_sampling_Planar_5DOF.png")

    # multiple inputs
    js = np.array([js, js])
    tcp = robot.forward(js)
