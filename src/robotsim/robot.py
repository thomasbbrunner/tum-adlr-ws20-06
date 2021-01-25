
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

    @abstractmethod
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
        pass

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
