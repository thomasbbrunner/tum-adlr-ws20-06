
import matplotlib.pyplot as plt
import numpy as np
import pdb
from torch.utils.data import Dataset

import robotsim


class RobotSimDataset(Dataset):
    """
    TODO:
    - More uniform distribution of samples in cartesian space.
    """

    # for generation of random values
    # according to new numpy documentation
    random_gen = np.random.default_rng()

    def __init__(self, robot, num_samples):
        """Dataset for robot simulation.

        Dataset is generated on demand. 
        Entire dataset is stored in memory.

        Args:
            robot: instance of RobotSim
            num_samples: number of samples in dataset.

        Example usage:
        >>> robot = robotsim.Robot2D3DoF([3, 3, 3])
        >>> dataset = RobotSimDataset(robot, 100)
        """

        self._robot = robot
        self._num_samples = int(num_samples)
        self._num_dof = self._robot.NUM_DOF

        joint_ranges = self._robot.get_joint_ranges()

        # sample random combinations of joint states
        self._joint_states = self.random_gen.uniform(
            joint_ranges[:, 0], joint_ranges[:, 1],
            (self._num_samples, self._num_dof))

        self._tcp_coords = robot.forward(self._joint_states)

        if self._tcp_coords.shape[0] != self._num_samples:
            raise RuntimeError(
                "Inconsistent sizes in dataset contents.")

    def __len__(self):
        """Returns number of samples in dataset.
        """
        return self._num_samples

    def __getitem__(self, item):
        """Returs tuple with joint states and TCP coordinates.
        """
        # exclude tcp orientation
        return self._joint_states[item], self._tcp_coords[item][:2]

    def plot(self, path=None, show=False):

        fig, ax = plt.subplots()
        ax.grid()
        ax.scatter(
            x=self._tcp_coords[:, 0],
            y=self._tcp_coords[:, 1],
            marker='.', s=1)

        ax.set_title(
            "{} DoF Dataset ({} samples)". format(
                self._num_dof, self._num_samples))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        if path:
            plt.savefig(path)

        if show:
            plt.show()

    def histogram(self, path=None, show=False, bins=200):

        fig, ax = plt.subplots()
        ax.grid()
        _, _, _, im = ax.hist2d(
            self._tcp_coords[:, 0],
            self._tcp_coords[:, 1],
            bins=bins,
            cmin=1,)

        ax.set_title(
            "Dataset {} DoF ({} samples)". format(
                self._num_dof, self._num_samples))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("samples")

        if path:
            plt.savefig(path)

        if show:
            plt.show()

    def plot_configurations(self, transparency=None, path=None, show=False):
        self._robot.plot_heatmap(
            self._joint_states,
            transparency, path, show)


if __name__ == "__main__":

    num_samples = 1e6

    robot = robotsim.Robot2D2DoF([3, 2])
    dataset = RobotSimDataset(robot, num_samples)
    dataset.plot(path="./figures/dataset_2D2DoF")
    dataset.histogram(path="./figures/dataset_2D2DoF_histogram")

    robot = robotsim.Robot2D3DoF([3, 2, 3])
    dataset = RobotSimDataset(robot, num_samples)
    dataset.plot(path="./figures/dataset_2D3DoF")
    dataset.histogram(path="./figures/dataset_2D3DoF_histogram")

    robot = robotsim.Robot2D4DoF([3, 2, 3])
    dataset = RobotSimDataset(robot, num_samples)
    dataset.plot(path="./figures/dataset_2D4DoF")
    dataset.histogram(path="./figures/dataset_2D4DoF_histogram")

    # plt.show()
