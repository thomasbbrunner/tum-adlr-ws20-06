
import matplotlib.pyplot as plt
import numpy as np
import pdb
from torch.utils.data import Dataset

import robotsim


class RobotSimDataset(Dataset):
    """TODO:
    - improve interface: 
        commenting out code to enable features is not ideal
        (e.g. joint ranges)
    """

    # for generation of random values
    # according to new numpy documentation
    random_gen = np.random.default_rng(seed=42)

    def __init__(self, robot, num_samples, normal=True):
        """Dataset for robot simulation.

        Dataset is generated on demand. 
        Entire dataset is stored in memory.

        Args:
            robot: instance of RobotSim.
            num_samples: number of samples in dataset.
            normal: sampled from normal distribution, 
                otherwise uniform distribution.

        Example usage:
        robot = robotsim.Robot2D3DoF([3, 3, 3])
        dataset = RobotSimDataset(robot, 100)
        """

        self._robot = robot
        self._num_samples = int(num_samples)
        self._num_dof = self._robot.NUM_DOF

        joint_ranges = self._robot.get_joint_ranges()

        # TODO: Check whether better
        if normal:
            # sample from random normal distribution N(0, std) with std=0.5
            self._joint_states = self.random_gen.normal(
                loc=0.0, scale=0.5, size=(self._num_samples, self._num_dof))
        else:
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
            "Dataset {} DoF ({} samples)". format(
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

    # 2 DoF
    robot = robotsim.Robot2D2DoF([3, 2])
    dataset = RobotSimDataset(robot, num_samples)
    # dataset.plot(path="./figures/dataset/normal_Robot2D2DoF")
    dataset.histogram(path="./figures/dataset/normal_Robot2D2DoF_histogram")

    dataset = RobotSimDataset(robot, num_samples, normal=False)
    # dataset.plot(path="./figures/dataset/uniform_Robot2D2DoF")
    dataset.histogram(path="./figures/dataset/uniform_Robot2D2DoF_histogram")

    # 3 DoF
    robot = robotsim.Robot2D3DoF([3, 2, 3])
    dataset = RobotSimDataset(robot, num_samples)
    # dataset.plot(path="./figures/dataset/normal_Robot2D3DoF")
    dataset.histogram(path="./figures/dataset/normal_Robot2D3DoF_histogram")

    dataset = RobotSimDataset(robot, num_samples, normal=False)
    # dataset.plot(path="./figures/dataset/uniform_Robot2D3DoF")
    dataset.histogram(path="./figures/dataset/uniform_Robot2D3DoF_histogram")

    # 4 DoF
    robot = robotsim.Robot2D4DoF([3, 2, 2, 3])
    dataset = RobotSimDataset(robot, num_samples)
    # dataset.plot(path="./figures/dataset/normal_Robot2D4DoF")
    dataset.histogram(path="./figures/dataset/normal_Robot2D4DoF_histogram")

    dataset = RobotSimDataset(robot, num_samples, normal=False)
    # dataset.plot(path="./figures/dataset/uniform_Robot2D4DoF")
    dataset.histogram(path="./figures/dataset/uniform_Robot2D4DoF_histogram")

    # Paper's Robot
    robot = robotsim.RobotPaper([3, 2, 3])
    dataset = RobotSimDataset(robot, num_samples)
    # dataset.plot(path="./figures/dataset/normal_RobotPaper")
    dataset.histogram(path="./figures/dataset/normal_RobotPaper_histogram")

    dataset = RobotSimDataset(robot, num_samples, normal=False)
    # dataset.plot(path="./figures/dataset/uniform_RobotPaper")
    dataset.histogram(path="./figures/dataset/uniform_RobotPaper_histogram")

    # plt.show()
