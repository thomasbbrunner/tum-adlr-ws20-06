
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pdb


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
    def get_joint_samples(self, num_samples, random=False):
        pass

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

    def plot(self, joint_states, path=None, separate_plots=True, show=False):
        """Plots robot for specified joint states.

        Also accepts batches of joint states.

        If a path is provided, the plot is saved to an image.

        Plot is only shown if "show" is set.

        Examples:
        # >>> # for single plot
        # >>> robot.plot_configurations([0, 1, 0])

        # >>> # for single plot with two configurations
        # >>> robot.plot([
                [0, 1, 0],
                [1, 0, 1]],
                separate_plots=False)

        # >>> # for 2 plots for each configuration
        # >>> robot.plot([
                [0, 1, 0],
                [1, 0, 1]])
        """

        joint_coords = self._get_joint_coords(joint_states)

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
            # TODO rethink plot limits
            robot_length = np.sum(self.len_links)*1.1
            ax.set_xlim([-robot_length, robot_length])
            ax.set_ylim([-robot_length, robot_length])

            # TODO improve performance
            # one call to plot -> add 'None' between arm configurations
            for arm in joint_coords:
                ax.plot(
                    arm[:, 0].flatten(),
                    arm[:, 1].flatten(),
                    c='b')

            ax.scatter(
                joint_coords[:, :, 0].flatten(),
                joint_coords[:, :, 1].flatten(),
                c='r', s=8)

        if path:
            plt.savefig(path)

        if show:
            plt.show()

    def plot_heatmap(self, joint_states, transparency=None, path=None, show=False):
        # TODO docstring
        # TODO fix examples in docstrings

        joint_coords = self._get_joint_coords(joint_states)

        fig, ax = plt.subplots()
        ax.grid()
        # TODO rethink plot limits
        robot_length = np.sum(self.len_links)*1.1
        ax.set_xlim([-robot_length, robot_length])
        ax.set_ylim([-robot_length, robot_length])

        if not transparency:
            # attempt at automatic transparency
            transparency = np.clip(
                np.exp(-0.008*joint_coords.shape[0]),
                0.01, 1)

        for arm in joint_coords:
            ax.plot(
                arm[:, 0].flatten(),
                arm[:, 1].flatten(),
                c='b', alpha=transparency)

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
    # >>> robot = Robot2DoF([3, 2])
    """

    NUM_DOF = 2

    def __init__(self, len_links):
        self.sim = Robot2D3DoF([*len_links, 0])
        self.len_links = self.sim.len_links

    def forward(self, joint_states, squeeze=True):

        joint_states = np.atleast_2d(joint_states)
        joint_states = np.hstack(
            (joint_states, np.zeros((joint_states.shape[0], 1))))

        return self.sim.forward(joint_states, squeeze)

    def inverse(self, tcp_coordinates, squeeze=True):

        joint_states = self.sim.inverse(tcp_coordinates, squeeze=False)

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

        return self.sim._get_joint_coords(joint_states)

    def get_joint_samples(self, num_samples, random=False):

        if random:
            joint_samples = self.random_gen.uniform(
                -np.pi, np.pi, (self.NUM_DOF, num_samples))

        else:
            joint_samples = np.linspace(
                -np.pi, np.pi, num_samples, endpoint=False)
            joint_samples = np.resize(
                joint_samples, (self.NUM_DOF, num_samples))

        return joint_samples


class Robot2D3DoF(RobotSim):
    """Simulation of 2D robotic arm with three revolute joints.

    Examples:
    # >>> robot = Robot3DoF([3, 2, 1])
    """

    NUM_DOF = 3

    def __init__(self, len_links):

        self.len_links = np.array(len_links)

        if self.len_links.shape[0] != self.NUM_DOF:
            raise RuntimeError(
                "Conflicting link lengths: {}". format(self.len_links))

        if not np.all(np.greater_equal(self.len_links, 0)):
            raise RuntimeError(
                "Link length has to be non-negative: {}". format(self.len_links))

    def forward(self, joint_states, squeeze=True):

        joint_states = np.atleast_2d(joint_states)

        if joint_states.shape[1] != self.NUM_DOF:
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

        theta = self.wrap(theta)
        theta = np.swapaxes(theta, 0, 2)

        if squeeze:
            theta = np.squeeze(theta)

        return theta

    def inverse_sampling(self, tcp_coordinates, step=None, num_samples=None, random=False):

        if step:
            tcp_angles = np.arange(-np.pi, np.pi, step)
        elif num_samples and not random:
            tcp_angles = np.linspace(
                -np.pi, np.pi, num_samples, endpoint=False)
        elif num_samples and random:
            tcp_angles = self.random_gen.uniform(
                -np.pi, np.pi, num_samples)

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
            0, self.len_links[0], 0, joint_states[:, 1], False)
        T03 = T02 @ self.dh_transformation(
            0, self.len_links[1], 0, joint_states[:, 2], False)
        Ttcp = T03 @ self.dh_transformation(
            0, self.len_links[2], 0, 0, False)

        # last column of T contains vector to each joint
        # dimensions: (num_inputs, joints, coords)
        joint_coords = np.swapaxes(
            np.array(
                [T01[:, :2, -1], T02[:, :2, -1], T03[:, :2, -1], Ttcp[:, :2, -1]]),
            0, 1)

        return joint_coords

    def get_joint_samples(self, num_samples, random=False):

        if random:
            joint_samples = self.random_gen.uniform(
                -np.pi, np.pi, (self.NUM_DOF, num_samples))

        else:
            joint_samples = np.linspace(
                -np.pi, np.pi, num_samples, endpoint=False)
            joint_samples = np.resize(
                joint_samples, (self.NUM_DOF, num_samples))

        return joint_samples


class Robot2D4DoF(RobotSim):
    """Simulation of 2D robotic arm with a prismatic and three revolute joints.

    Examples:
    # >>> robot = Robot2D4DoF([3, 3, 3])
    """

    NUM_DOF = 4

    def __init__(self, len_links):

        self.sim = Robot2D3DoF(len_links)

        self.length = np.sum(len_links)
        self.len_links = self.sim.len_links

    def forward(self, joint_states, squeeze=True):

        joint_states = np.atleast_2d(joint_states)

        tcp_coordinates = self.sim.forward(joint_states[:, 1:], squeeze=False)
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

        if step:
            tcp_angles = np.arange(-np.pi, np.pi, step)
            base_heights = np.arange(-self.length, self.length, step)

        elif num_samples and not random:
            tcp_angles = np.linspace(
                -np.pi, np.pi, num_samples, endpoint=False)
            base_heights = np.linspace(
                -self.length, self.length, num_samples)

        elif num_samples and random:
            tcp_angles = self.random_gen.uniform(
                -np.pi, np.pi, num_samples)
            base_heights = self.random_gen.uniform(
                -self.length, self.length, num_samples)

        tcp_coordinates = np.atleast_2d(tcp_coordinates)
        tcp_angles = np.repeat(tcp_angles, base_heights.shape[0])
        base_heights = np.resize(base_heights, tcp_angles.shape[0])
        sampling_vars = np.vstack((tcp_angles, base_heights)).T

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

        joint_states = self.sim.inverse(tcp_coordinates)
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

        joint_coords = self.sim._get_joint_coords(joint_states[:, 1:])
        joint_coords[:, :, 1] += np.reshape(joint_states[:, 0], (-1, 1))

        return joint_coords

    def get_joint_samples(self, num_samples, random=False):

        if random:
            joint_samples = np.vstack((
                self.random_gen.uniform(
                    -self.length, self.length, (1, num_samples)),
                self.random_gen.uniform(
                    -np.pi, np.pi, (3, num_samples))))

        else:
            joint_samples = np.vstack((
                np.linspace(-self.length, self.length, num_samples),
                np.resize(
                    np.linspace(-np.pi, np.pi, num_samples, endpoint=False),
                    (3, num_samples))))

        return joint_samples


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
    robot.plot(js, separate_plots=True)

    tcp = robot.forward(js)
    js_inv = robot.inverse(tcp)
    robot.plot(js_inv, separate_plots=False)

    tcp = robot.forward([js, js])
    js_inv = robot.inverse(tcp)
    robot.plot(js_inv, separate_plots=False)

    # 3 DoF
    js = [1, -2.2, 0.4]
    robot = Robot2D3DoF([3, 2, 3])
    robot.plot(js, separate_plots=True)

    tcp = robot.forward(js)
    js_inv = robot.inverse(tcp)
    robot.plot(js_inv, separate_plots=False)
    js_sam = robot.inverse_sampling(tcp, num_samples=500)
    robot.plot_heatmap(
        js_sam, path="./figures/robotsim_inverse_sampling_2D3DoF", transparency=0.09)

    tcp = robot.forward([js, js])
    js_inv = robot.inverse(tcp)
    robot.plot(js_inv, separate_plots=False)
    js_sam = robot.inverse_sampling(tcp, num_samples=100)
    robot.plot(js_sam, separate_plots=False)

    # 4 DoF
    js = [2.1, 1, -2.2, 0.4]
    robot = Robot2D4DoF([3, 2, 3])
    robot.plot(js, separate_plots=True)

    tcp = robot.forward(js)
    js_sam = robot.inverse_sampling(tcp, num_samples=150)
    robot.plot_heatmap(
        js_sam, path="./figures/robotsim_inverse_sampling_2D4DoF", transparency=0.002)

    tcp = robot.forward([js, js])
    js_sam = robot.inverse_sampling(tcp, num_samples=50)
    robot.plot(js_sam, separate_plots=False)

    # plots
    # plt.show()
