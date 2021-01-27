
from abc import ABC, abstractmethod
import numpy as np


class Robot(ABC):
    """Abstract class for robot simulations.
    """

    # for generation of random values
    # according to new numpy documentation
    random_gen = np.random.default_rng()

    @abstractmethod
    def forward(self, joint_states, squeeze=True):
        """Returns TCP coordinates for specified joint states.
        Also accepts batch processing of several joint states.

        Args:
        joint_states: state of each joint.
        squeeze: remove single-dimensional entries from output array.

        Returns:
        tcp_coordinates: (x, y, phi) coordinates of TCP.

        Examples:
        # for a single state and a robot with three joints
        robot.forward(
            [0, 1, 0])

        # for batch of two states and a robot with three joints
        robot.forward([
            [0, 1, 0],
            [1, 0, 1]])
        """
        pass

    @abstractmethod
    def inverse(self, tcp_coordinates, squeeze=True):
        """Returns joint states for specified TCP coordinates.
        Also accepts batch processing of several TCP coordinates.
        Returns at most two possible solutions for each input.
        If no solution is found, np.nan is returned.

        Args:
        tcp_coordinates: (x, y, phi) coordinates of TCP.
        squeeze: remove single-dimensional entries from output array.

        Returns:
        joint_states: state of each joint.

        Examples:
        # for a single coordinate
        robot.inverse([3, -4, 0])

        # for batch of two coordinates
        robot.inverse([
            [3, -4, 0],
            [5, 1, 0.3]])
        """
        pass

    def rejection_sampling(
            self, tcp_coordinates, num_samples, eps, mean, std):
        """Samples possible robot configurations 
        that reach the specified TCP.

        Args:
        tcp_coordinates: (x, y, phi) coordinates of TCP.
        num_samples: number of configurations that are returned.
        eps: accuracy of sampling 
            (how far can the sampled TCP coordinates be 
            from the given coordinates).
        mean: mean of the normal distribution for sampling.
        std: standard deviation of the normal distribution for sampling.

        Returns:
        joint_states: state of each joint.
        """

        tcp_coordinates = np.atleast_2d(tcp_coordinates)

        if tcp_coordinates.shape[1] != 2:
            raise RuntimeError(
                "Wrong shape of TCP coordinates: {}"
                .format(tcp_coordinates))

        if np.linalg.norm(tcp_coordinates) > self.get_length():
            raise RuntimeError(
                "TCP coordinates are outside workspace: {}"
                .format(tcp_coordinates))

        joint_states = np.zeros((num_samples, self.get_dof()))
        hit_samples = 0

        # how many samples to generate in each loop
        # generating many samples per loop is more efficient
        batch_size = num_samples*100

        while hit_samples < num_samples:

            sampled_joints = self.random_gen.normal(
                loc=mean, scale=std, size=(batch_size, self.get_dof()))
            sampled_tcp = self.forward(sampled_joints, squeeze=False)[:, :2]

            distances = np.linalg.norm(sampled_tcp - tcp_coordinates, axis=1)

            # select samples that are close to target
            sampled_joints = sampled_joints[distances <= eps]

            # add samples to output array
            lo_index = hit_samples
            hi_index = np.minimum(
                lo_index + sampled_joints.shape[0], num_samples)
            joint_states[lo_index:hi_index] = sampled_joints[:hi_index-lo_index]
            hit_samples += sampled_joints.shape[0]

        return joint_states

    @abstractmethod
    def get_joint_coords(self, joint_states):
        """Calculates coordinates for each joint center 
        and for the TCP.
        Also accepts batch processing of several joint states.

        Args:
        joint_states: state of each joint.

        Returns:
        joint_coords: coordinates of joints and of the TCP.
        """
        pass

    @abstractmethod
    def get_joint_ranges(self):
        """Returns ranges of each joint.
        """
        pass

    @abstractmethod
    def get_length(self):
        """Returns length of robot (sum of length of all links).
        """
        pass

    @abstractmethod
    def get_dof(self):
        """Returns number of DOF of robot.
        """
        pass
