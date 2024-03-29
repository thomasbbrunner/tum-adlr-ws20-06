
raise RuntimeError("This module is deprecated!")

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pdb

import robotsim_plot


class RobotSim(ABC):

    # for generation of random values
    # according to new numpy documentation
    random_gen = np.random.default_rng()

    @abstractmethod
    def forward(self, joint_states):
        """Returns TCP coordinates for specified joint states.

        TCP coordinates have format (x, y, phi)
            x,y:  coordinates of TCP
            phi:  angle of TCP with respect to
                  horizontal in range [-pi, pi).

        Also accepts batch processing of several joint states.

        Examples:
        # >>> # for a single state and a robot with three joints
        # >>> robot.forward(
                [0, 1, 0])

        # >>> # for batch of two states and a robot with three joints
        # >>> robot.forward([
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

        Examples:
        # >>> # for a single coordinate
        # >>> robot.inverse([3, -4, 0])
        #
        # >>> # for batch of two coordinates
        # >>> robot.inverse([
                [3, -4, 0],
                [5, 1, 0.3]])
        """
        pass

    @abstractmethod
    def inverse_sampling(self, tcp_coordinates, step=None, num_samples=None, random=False):
        """Generates samples of inverse kinematics
        based on (x, y) TCP coordinates.

        You must provide either step or num_samples, but not both!
        "step" and "num_samples" are applied to all sampling variables.

        Also accepts batch processing of several TCP coordinates.
        In this case, step and num_samples are applied to
        each set of TCP coordinates.

        Based on approximate Bayesian computation.

        Args:
            tcp_coordinates: (x, y) coordinates of TCP.
            step: specifies the step size between samples.
            num_samples: specifies number of samples to generate.
            random: pick samples randomly or in fix distances.

        Examples:
        # >>> # for a single coordinate and step
        # >>> robot.full_inverse([7, 3], step=0.1)
        #
        # >>> # for a single coordinate and number of samples
        # >>> robot.full_inverse([7, 3], num_samples=20)
        #
        # >>> # for batch of three coordinates and total of 300 samples
        # >>> robot.full_inverse(
                [[7, 3], [-7, -3], [7, -3]], num_samples=100)
        """
        pass

    @abstractmethod
    def _get_joint_coords(self, joint_states):
        pass

    @abstractmethod
    def get_joint_ranges(self):
        pass

    def get_length(self):
        return self._length

    @staticmethod
    def wrap(angles):
        """Wraps angles to [-pi, pi) range."""
        # wrap angles to range [-pi, pi)
        return (angles + np.pi) % (2*np.pi) - np.pi

        # wrap angles to range [0, 2*pi)
        # return angles % (2*np.pi)

    @staticmethod
    def dh_transformation(alpha, a, d, theta, squeeze=True):
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

        if squeeze:
            transformation = np.squeeze(transformation)

        return transformation


class Robot3D1(RobotSim):
    # create example of 3d with 3dof
    # http://courses.csail.mit.edu/6.141/spring2011/pub/lectures/Lec14-Manipulation-II.pdf (slide 29)
    pass


class Robot2D2DoF(RobotSim):
    """Simulation of 2D robotic arm with two revolute joints.

    Examples:
    # >>> robot = Robot2DoF([3, 2])
    """

    NUM_DOF = 2

    def __init__(self, len_links):
        self._sim = Robot2D3DoF([*len_links, 0])
        self._len_links = self._sim._len_links
        self._length = np.sum(self._len_links)

    def forward(self, joint_states, squeeze=True):

        joint_states = np.atleast_2d(joint_states)
        joint_states = np.hstack(
            (joint_states, np.zeros((joint_states.shape[0], 1))))

        return self._sim.forward(joint_states, squeeze)

    def inverse(self, tcp_coordinates, squeeze=True):

        joint_states = self._sim.inverse(tcp_coordinates, squeeze=False)

        # remove solutions for joint 3
        joint_states = np.atleast_3d(joint_states)[:, :, :2]

        if squeeze:
            joint_states = np.squeeze(joint_states)

        return joint_states

    def inverse_sampling(self):
        raise RuntimeError(
            "This function cannot be implemented.")

    def _get_joint_coords(self, joint_states):

        joint_states = np.reshape(joint_states, (-1, self.NUM_DOF))

        if joint_states.shape[1] != self.NUM_DOF:
            raise RuntimeError(
                "Expected different size for joint_states: {}".format(joint_states))

        joint_states = np.hstack(
            (joint_states, np.zeros((joint_states.shape[0], 1))))

        return self._sim._get_joint_coords(joint_states)

    def get_joint_ranges(self):

        return np.array([
            [-np.pi, np.pi],
            [-np.pi, np.pi],
        ])

        # return np.array([
        #     [-np.pi/2, np.pi/2],
        #     [-np.pi/2, np.pi/2],
        # ])


class Robot2D3DoF(RobotSim):
    """Simulation of 2D robotic arm with three revolute joints.

    Examples:
    # >>> robot = Robot3DoF([3, 2, 1])
    """

    NUM_DOF = 3

    def __init__(self, len_links):

        self._len_links = np.array(len_links)
        self._length = np.sum(self._len_links)

        if self._len_links.shape[0] != self.NUM_DOF:
            raise RuntimeError(
                "Conflicting link lengths: {}". format(self._len_links))

        if not np.all(np.greater_equal(self._len_links, 0)):
            raise RuntimeError(
                "Link length has to be non-negative: {}". format(self._len_links))

    def forward(self, joint_states, squeeze=True):

        joint_states = np.atleast_2d(joint_states)

        if joint_states.shape[1] != self.NUM_DOF:
            raise RuntimeError(
                "Conflicting joint states: {}". format(joint_states))

        theta = np.zeros((joint_states.shape[0], 3))
        theta[:, :joint_states.shape[1]] = joint_states

        tcp_coordinates = np.array([
            # x-coordinate of TCP
            self._len_links[0]*np.cos(theta[:, 0]) +
            self._len_links[1]*np.cos(theta[:, 0] + theta[:, 1]) +
            self._len_links[2]*np.cos(theta[:, 0] + theta[:, 1] + theta[:, 2]),
            # y-coordinate of TCP
            self._len_links[0]*np.sin(theta[:, 0]) +
            self._len_links[1]*np.sin(theta[:, 0] + theta[:, 1]) +
            self._len_links[2]*np.sin(theta[:, 0] + theta[:, 1] + theta[:, 2]),
            # angle of TCP with respect to horizontal
            theta[:, 0] + theta[:, 1] + theta[:, 2]]).T

        tcp_coordinates[:, 2] = self.wrap(tcp_coordinates[:, 2])

        if squeeze:
            tcp_coordinates = np.squeeze(tcp_coordinates)

        return tcp_coordinates

    def inverse(self, tcp_coordinates, squeeze=True):

        tcp_coordinates = np.atleast_2d(tcp_coordinates)

        if tcp_coordinates.shape[1] != 3:
            raise RuntimeError(
                "Missing TCP coordinates information: {}". format(tcp_coordinates))

        xtcp = tcp_coordinates[:, 0]
        ytcp = tcp_coordinates[:, 1]
        phi = tcp_coordinates[:, 2]
        l1 = self._len_links[0]
        l2 = self._len_links[1]
        l3 = self._len_links[2]

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

        theta = self.wrap(theta)
        theta = np.swapaxes(theta, 0, 2)

        if squeeze:
            theta = np.squeeze(theta)

        return theta

    def inverse_sampling(self, tcp_coordinates, step=None, num_samples=None, random=False):

        if ((step is None and num_samples is None) or
                (step is not None and num_samples is not None)):
            raise RuntimeError(
                "Please provide either a step or a num_samples value.")

        if step:
            tcp_angles = np.arange(-np.pi, np.pi, step)
        elif num_samples and not random:
            tcp_angles = np.linspace(
                -np.pi, np.pi, num_samples, endpoint=False)
        elif num_samples and random:
            tcp_angles = self.random_gen.uniform(
                -np.pi, np.pi, num_samples)
        else:
            raise RuntimeError(
                "Unsupported combination of input parameters.")

        tcp_coordinates = np.repeat(
            np.atleast_2d(tcp_coordinates),
            tcp_angles.shape[0],
            axis=0)

        tcp_angles = np.resize(tcp_angles, (tcp_coordinates.shape[0], 1))
        tcp_coordinates = np.hstack((tcp_coordinates[:, :2], tcp_angles))

        joint_states = self.inverse(tcp_coordinates)

        # remove invalid solutions
        joint_states = joint_states[~np.isnan(joint_states)]
        joint_states = np.reshape(joint_states, (-1, 3))

        return joint_states

    def _get_joint_coords(self, joint_states):

        joint_states = np.reshape(joint_states, (-1, self.NUM_DOF))

        T01 = self.dh_transformation(
            0, 0, 0, joint_states[:, 0], False)
        T02 = T01 @ self.dh_transformation(
            0, self._len_links[0], 0, joint_states[:, 1], False)
        T03 = T02 @ self.dh_transformation(
            0, self._len_links[1], 0, joint_states[:, 2], False)
        Ttcp = T03 @ self.dh_transformation(
            0, self._len_links[2], 0, 0, False)

        # last column of T contains vector to each joint
        # dimensions: (num_inputs, joints, coords)
        joint_coords = np.swapaxes(
            np.array(
                [T01[:, :2, -1], T02[:, :2, -1], T03[:, :2, -1], Ttcp[:, :2, -1]]),
            0, 1)

        return joint_coords

    def get_joint_ranges(self):

        return np.array([
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
        ])

        # return np.array([
        #     [-np.pi/2, np.pi/2],
        #     [-np.pi/2, np.pi/2],
        #     [-np.pi/2, np.pi/2],
        # ])


class RobotPaper(RobotSim):
    """Simulation of 2D robotic arm with a prismatic and three revolute joints.

    Examples:
    # >>> robot = Robot2D4DoF([3, 3, 3])
    """

    NUM_DOF = 4

    def __init__(self, len_links):

        self._sim = Robot2D3DoF(len_links)

        self._length = np.sum(self._sim._len_links)

    def forward(self, joint_states, squeeze=True):

        joint_states = np.atleast_2d(joint_states)

        tcp_coordinates = self._sim.forward(joint_states[:, 1:], squeeze=False)
        tcp_coordinates[:, 1] += joint_states[:, 0]  # add base height

        if squeeze:
            tcp_coordinates = np.squeeze(tcp_coordinates)

        return tcp_coordinates

    def inverse(self):
        raise RuntimeError(
            "This function cannot be implemented.")

    def inverse_sampling(self, tcp_coordinates, step=None, num_samples=None, random=False):

        if ((step is None and num_samples is None) or
                (step is not None and num_samples is not None)):
            raise RuntimeError(
                "Please provide either a step or a num_samples value.")

        tcp_coordinates = np.atleast_2d(tcp_coordinates)
        tcp_coordinates = tcp_coordinates[:, :2]  # remove any unneeded values

        if not random:
            if step:
                tcp_angles = np.arange(-np.pi, np.pi, step)
                base_heights = np.arange(-self._length, self._length, step)

            elif num_samples and not random:
                tcp_angles = np.linspace(
                    -np.pi, np.pi, num_samples, endpoint=False)
                base_heights = np.linspace(
                    -self._length, self._length, num_samples)

            tcp_angles = np.repeat(tcp_angles, base_heights.shape[0])
            base_heights = np.resize(base_heights, tcp_angles.shape[0])
            sampling_vars = np.vstack((tcp_angles, base_heights)).T

        elif random and num_samples:
            sampling_vars = self.random_gen.uniform(
                [-np.pi,  -self._length], [np.pi,  self._length],
                (num_samples**2, 2))

        else:
            raise RuntimeError(
                "Unsupported combination of input parameters.")

        tcp_coordinates = np.repeat(
            tcp_coordinates,
            sampling_vars.shape[0],
            axis=0)
        sampling_vars = np.resize(
            sampling_vars,
            (tcp_coordinates.shape[0], 2))

        # The idea here is to move tcp coordinate to the x-axis.
        # We ignore the original tcp y-coordinates and
        # replace them by the base heights we sampled.
        # It simplifies finding solution and
        # allows using the inverse kinematics of the 3 DoF robot.
        tcp_shift = tcp_coordinates[:, 1].copy()
        tcp_coordinates = np.vstack(
            (tcp_coordinates[:, 0], sampling_vars[:, 1], sampling_vars[:, 0])).T

        joint_states = self._sim.inverse(tcp_coordinates)
        joint_states = np.reshape(joint_states, (-1, 3))

        # add tcp shift and base height back into states
        base_heights = np.reshape(
            np.repeat(
                tcp_shift - sampling_vars[:, 1],
                2),  # there are two solutions for each tcp coordinate
            (-1, 1))
        joint_states = np.hstack((base_heights, joint_states))

        # remove invalid solutions
        joint_states = joint_states[~np.isnan(joint_states).any(axis=1)]

        # TODO check duplicate solutions
        # tcp_coords input [1, 2]

        return joint_states

    def _get_joint_coords(self, joint_states):

        joint_states = np.reshape(joint_states, (-1, self.NUM_DOF))

        joint_coords = self._sim._get_joint_coords(joint_states[:, 1:])
        joint_coords[:, :, 1] += np.reshape(joint_states[:, 0], (-1, 1))

        return joint_coords

    def get_joint_ranges(self):

        return np.array([
            [-self._length, self._length],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
        ])

        # return np.array([
        #     [-1.0, 1.0],
        #     [-np.pi, np.pi],
        #     [-np.pi, np.pi],
        #     [-np.pi, np.pi],
        # ])


class Robot2D4DoF(RobotSim):
    """Simulation of 2D robotic arm with four revolute joints.

    Examples:
    # >>> robot = Robot2D4DoF([3, 3, 3, 3])
    """

    NUM_DOF = 4

    def __init__(self, len_links):

        self._len_links = len_links
        self._sim = Robot2D3DoF(self._len_links[:3])
        self._length = np.sum(self._len_links)

    def forward(self, joint_states, squeeze=True):

        joint_states = np.atleast_2d(joint_states)

        if joint_states.shape[1] != self.NUM_DOF:
            raise RuntimeError(
                "Expected input of a different size: {}". format(joint_states))

        theta = joint_states

        tcp_coordinates = np.array([
            # x-coordinate of TCP
            self._len_links[0]*np.cos(theta[:, 0]) +
            self._len_links[1]*np.cos(theta[:, 0] + theta[:, 1]) +
            self._len_links[2]*np.cos(theta[:, 0] + theta[:, 1] + theta[:, 2]) +
            self._len_links[3]*np.cos(
                theta[:, 0] + theta[:, 1] + theta[:, 2] + theta[:, 3]),
            # y-coordinate of TCP
            self._len_links[0]*np.sin(theta[:, 0]) +
            self._len_links[1]*np.sin(theta[:, 0] + theta[:, 1]) +
            self._len_links[2]*np.sin(theta[:, 0] + theta[:, 1] + theta[:, 2]) +
            self._len_links[3]*np.sin(
                theta[:, 0] + theta[:, 1] + theta[:, 2] + theta[:, 3]),
            # angle of TCP with respect to horizontal
            theta[:, 0] + theta[:, 1] + theta[:, 2] + theta[:, 3]]).T

        tcp_coordinates[:, 2] = self.wrap(tcp_coordinates[:, 2])

        if squeeze:
            tcp_coordinates = np.squeeze(tcp_coordinates)

        return tcp_coordinates

    def inverse(self):
        raise RuntimeError(
            "This function cannot be implemented.")

    def inverse_sampling(self, tcp_coordinates, step=None, num_samples=None, random=False):

        if ((step is None and num_samples is None) or
                (step is not None and num_samples is not None)):
            raise RuntimeError(
                "Please provide either a step or a num_samples value.")

        tcp_coordinates = np.atleast_2d(tcp_coordinates)
        tcp_coordinates = tcp_coordinates[:, :2]  # remove any unneeded values

        if not random:
            if step:
                tcp_angles = np.arange(-np.pi, np.pi, step)
                link3_angles = np.arange(-np.pi, np.pi, step)

            elif num_samples:
                tcp_angles = np.linspace(
                    -np.pi, np.pi, num_samples, endpoint=False)
                link3_angles = np.linspace(
                    -np.pi, np.pi, num_samples, endpoint=False)

            tcp_angles = np.repeat(tcp_angles, link3_angles.shape[0])
            link3_angles = np.resize(link3_angles, tcp_angles.shape[0])

        elif random and num_samples:
            tcp_angles = self.random_gen.uniform(-np.pi, np.pi, num_samples**2)
            link3_angles = self.random_gen.uniform(
                -np.pi, np.pi, num_samples**2)

        else:
            raise RuntimeError(
                "Unsupported combination of input parameters.")

        assert tcp_angles.shape == link3_angles.shape

        tcp_coordinates = np.repeat(
            tcp_coordinates,
            tcp_angles.shape[0],
            axis=0)
        tcp_angles = np.resize(
            tcp_angles,
            (tcp_coordinates.shape[0], 1))
        link3_angles = np.resize(
            link3_angles,
            (tcp_coordinates.shape[0], 1))

        # TODO optimize this
        tcp_angles_trig = np.hstack((
            np.cos(tcp_angles), np.sin(tcp_angles)))

        # for each tcp coordinate and tcp angle pair
        # calculate the origin of the third joint
        # to use the inverse kinematics equations
        joint4_coordinates = tcp_coordinates + \
            self._len_links[3]*tcp_angles_trig

        # contains (x_4, y_4 and theta_4)
        joint4_coordinates = np.hstack(
            (joint4_coordinates, link3_angles))

        joint_states = self._sim.inverse(joint4_coordinates)
        joint_states = np.reshape(joint_states, (-1, 3))
        joint4_angles = np.reshape(
            np.repeat(tcp_angles - link3_angles + np.pi, 2), (-1, 1))
        joint_states = np.hstack((
            joint_states, joint4_angles))

        # remove entries that did not have valid inverse solution
        joint_states = joint_states[~np.isnan(joint_states).any(axis=1)]

        return joint_states

    def _get_joint_coords(self, joint_states):

        joint_states = np.reshape(joint_states, (-1, self.NUM_DOF))

        T01 = self.dh_transformation(
            0, 0, 0, joint_states[:, 0], False)
        T02 = T01 @ self.dh_transformation(
            0, self._len_links[0], 0, joint_states[:, 1], False)
        T03 = T02 @ self.dh_transformation(
            0, self._len_links[1], 0, joint_states[:, 2], False)
        T04 = T03 @ self.dh_transformation(
            0, self._len_links[2], 0, joint_states[:, 3], False)
        Ttcp = T04 @ self.dh_transformation(
            0, self._len_links[3], 0, 0, False)

        # last column of T contains vector to each joint
        # dimensions: (num_inputs, joints, coords)
        joint_coords = np.swapaxes(
            np.array(
                [T01[:, :2, -1], T02[:, :2, -1],
                 T03[:, :2, -1], T04[:, :2, -1], Ttcp[:, :2, -1]]),
            0, 1)

        return joint_coords

    def get_joint_ranges(self):

        return np.array([
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
        ])


if __name__ == "__main__":

    # TODO: functionality tests
    # (1) difficult joint_states
    # (2) do forward pass
    # (3) do inverse pass
    # (4) check if same joint_states
    # (5) compare with reference values

    # 2 DoF
    js = [1, -2.2]
    robot = Robot2D2DoF([3, 2])
    robotsim_plot.plot(js, robot)

    tcp = robot.forward(js)
    js_inv = robot.inverse(tcp)
    robotsim_plot.plot(js_inv, robot)

    tcp = robot.forward([js, js])
    js_inv = robot.inverse(tcp)
    robotsim_plot.plot(js_inv, robot)

    # 3 DoF
    js = [1, -2.2, 0.4]
    robot = Robot2D3DoF([3, 2, 3])
    robotsim_plot.plot(js, robot)

    tcp = robot.forward(js)
    js_inv = robot.inverse(tcp)
    robotsim_plot.plot(js_inv, robot)
    js_sam = robot.inverse_sampling(tcp, num_samples=500)
    robotsim_plot.heatmap(
        js_sam, robot, path="./figures/dataset/inverse_sampling_Robot2D3DoF", transparency=0.09)
    js_sam = robot.inverse_sampling(tcp, num_samples=500, random=True)
    robotsim_plot.heatmap(
        js_sam, robot, path="./figures/dataset/inverse_sampling_Robot2D3DoF_random", transparency=0.09)

    tcp = robot.forward([js, js])
    js_inv = robot.inverse(tcp)
    robotsim_plot.plot(js_inv, robot)
    js_sam = robot.inverse_sampling(tcp, num_samples=100)
    robotsim_plot.plot(js_sam, robot)

    # 4 DoF
    js = [0.9, -1.3, 0.5, -0.5]
    robot = Robot2D4DoF([3, 2, 2, 3])
    robotsim_plot.plot(js, robot)

    tcp = robot.forward(js)
    js_sam = robot.inverse_sampling(tcp, num_samples=150)
    robotsim_plot.heatmap(
        js_sam, robot, path="./figures/dataset/inverse_sampling_Robot2D4DoF", transparency=0.002)
    js_sam = robot.inverse_sampling(tcp, num_samples=150, random=True)
    robotsim_plot.heatmap(
        js_sam, robot, path="./figures/dataset/inverse_sampling_Robot2D4DoF_random", transparency=0.002)

    tcp = robot.forward([js, js])
    js_sam = robot.inverse_sampling(tcp, num_samples=50)
    robotsim_plot.plot(js_sam, robot)

    # Paper's Robot
    js = [2.1, 1, -2.2, 0.4]
    robot = RobotPaper([3, 2, 3])
    robotsim_plot.plot(js, robot)

    tcp = robot.forward(js)
    js_sam = robot.inverse_sampling(tcp, num_samples=150)
    robotsim_plot.heatmap(
        js_sam, robot, path="./figures/dataset/inverse_sampling_RobotPaper", transparency=0.002)
    js_sam = robot.inverse_sampling(tcp, num_samples=150, random=True)
    robotsim_plot.heatmap(
        js_sam, robot, path="./figures/dataset/inverse_sampling_RobotPaper_random", transparency=0.002)

    tcp = robot.forward([js, js])
    js_sam = robot.inverse_sampling(tcp, num_samples=50)
    robotsim_plot.plot(js_sam, robot)

    # plt.show()
