
from abc import ABC, abstractmethod
import numpy as np

from robotsim.planar import Planar
from robotsim.plot import plot, heatmap
from robotsim.robot import Robot
import utils


class Paper(Robot):
    """Simulation of robot with one prismatic and three revolute joints
    (as proposed in the INN paper).
    For documentation, please refer to the base class.
    """

    def __init__(self, num_dof, len_links):

        self._num_dof = num_dof
        self._robot = Planar(num_dof-1, len_links)

    def forward(self, joint_states, squeeze=True):

        joint_states = np.atleast_2d(joint_states)

        tcp_coordinates = self._robot.forward(
            joint_states[:, 1:], squeeze=False)
        tcp_coordinates[:, 1] += joint_states[:, 0]  # add base height

        if squeeze:
            tcp_coordinates = np.squeeze(tcp_coordinates)

        return tcp_coordinates

    def inverse(self, tcp_coordinates, squeeze=True):

        raise RuntimeError(
            "Function not implemented.")

    def get_joint_coords(self, joint_states):

        joint_states = np.atleast_2d(joint_states)

        if joint_states.shape[1] != self._num_dof:
            raise RuntimeError(
                "Wrong shape of joint states: {}"
                .format(joint_states))

        joint_coords = self._robot.get_joint_coords(joint_states[:, 1:])
        joint_coords[:, :, 1] += np.reshape(joint_states[:, 0], (-1, 1))

        return joint_coords

    def get_joint_ranges(self):
        ranges = np.vstack(
            (np.array([-self.get_length(), self.get_length()]),
             self._robot.get_joint_ranges()))
        return ranges

    def get_length(self):
        return self._robot.get_length()

    def get_dof(self):
        return self._num_dof


if __name__ == "__main__":

    # functionality tests
    robot = Paper(5, [1, 2, 1, 2])

    # one input
    js = [0, 1, -1, 1, -1]
    # plot(js, robot, show=True)
    tcp = robot.forward(js)
    js_sampling = robot.rejection_sampling(
        tcp[:2], num_samples=5000, eps=0.05, mean=0, std=1)
    heatmap(
        js_sampling, robot, highlight=1, transparency=0.005,
        path="./figures/robotsim/rejection_sampling_Paper_5DOF.png")

    # multiple inputs
    js = np.array([js, js])
    tcp = robot.forward(js)
