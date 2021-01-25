
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

    def __init__(self, len_links):

        self._num_dof = 4
        self._robot = Planar(3, len_links)

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

    def rejection_sampling(self, tcp_coordinates, num_samples, eps, mean, std):

        tcp_coordinates = np.atleast_2d(tcp_coordinates)

        if tcp_coordinates.shape[1] != 2:
            raise RuntimeError(
                "Wrong shape of TCP coordinates: {}"
                .format(tcp_coordinates))

        if np.linalg.norm(tcp_coordinates) >= self.get_length():
            raise RuntimeError(
                "TCP coordinates are outside workspace: {}"
                .format(tcp_coordinates))

        joint_states = np.zeros((num_samples, self._num_dof))
        hit_samples = 0

        batch_size = num_samples*100

        while hit_samples < num_samples:

            sampled_joints = self.random_gen.normal(
                loc=mean, scale=std, size=(batch_size, self._num_dof))
            sampled_tcp = self.forward(sampled_joints, squeeze=False)[:, :2]

            distances = np.linalg.norm(sampled_tcp - tcp_coordinates, axis=1)

            sampled_joints = sampled_joints[distances <= eps]

            lo_index = hit_samples
            hi_index = np.minimum(
                lo_index + sampled_joints.shape[0], num_samples)
            joint_states[lo_index:hi_index] = sampled_joints[:hi_index-lo_index]
            hit_samples += sampled_joints.shape[0]

        return joint_states

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
    robot = Paper([1, 1, 1])

    # one input
    js = [0, 1, -1, 1]
    plot(js, robot, show=True)
    tcp = robot.forward(js)
    js_sampling = robot.rejection_sampling(
        tcp[:2], num_samples=1000, eps=0.05, mean=0, std=1)
    heatmap(js_sampling, robot, show=True)

    # multiple inputs
    js = np.array([js, js])
    tcp = robot.forward(js)
