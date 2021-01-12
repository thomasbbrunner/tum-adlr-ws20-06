
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
            # TODO 0.5 is standard deviation (not variance!)
            self._joint_states = self.random_gen.normal(
                loc=0.0, scale=0.5, size=(self._num_samples, self._num_dof))
        else:
            # TODO reset joint limits to normal limits
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
            "TCP Coordinates in Dataset ({} DOF, {} samples)". format(
                self._num_dof, self._num_samples))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        if path:
            fig.savefig(path)

        if show:
            fig.show()

    def histogram(self, path=None, show=False, bins=150):

        fig, ax = plt.subplots()
        ax.grid()
        _, _, _, im = ax.hist2d(
            self._tcp_coords[:, 0],
            self._tcp_coords[:, 1],
            bins=bins,
            cmin=1,)

        ax.set_title(
            "TCP Coordinates in Dataset ({} DOF, {} samples)". format(
                self._num_dof, self._num_samples))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("samples")

        if path:
            fig.savefig(path, dpi=300)

        if show:
            fig.show()

    def plot_configurations(self, transparency=None, path=None, show=False):

        fig, ax = self._robot.plot_heatmap(
            self._joint_states,
            transparency, path=None, show=False)

        # get a sample to plot
        joint_coords = self._robot._get_joint_coords(self._joint_states[0])

        for arm in joint_coords:

            ax.plot(
                arm[:, 0].flatten(),
                arm[:, 1].flatten(),
                c='k', linewidth=2, zorder=10)

            ax.scatter(
                arm[:, 0].flatten(),
                arm[:, 1].flatten(),
                c='r', s=6, zorder=11)

        ax.set_title(
            "Configurations in Dataset ({} DOF, {} samples)". format(
                self._num_dof, self._num_samples))

        if path:
            fig.savefig(path, dpi=300)

        if show:
            fig.show()


if __name__ == "__main__":

    num_samples = 1e6
    num_samples_limited = 1e4  # for plots that can't handle full dataset

    # robots used during training
    robot_2dof = robotsim.Robot2D2DoF([0.5, 1])
    robot_3dof = robotsim.Robot2D3DoF([0.5, 0.5, 1.0])
    robot_4dof = robotsim.Robot2D4DoF([0.5, 0.5, 0.5, 1.0])
    robot_paper = robotsim.RobotPaper([0.5, 0.5, 1.0])

    robots = [robot_2dof, robot_3dof, robot_4dof, robot_paper]
    names = ["2dof", "3dof", "4dof", "paper"]

    # generation of dataset plots
    for robot, name in zip(robots, names):

        # normal
        dataset = RobotSimDataset(robot, num_samples_limited)
        dataset.plot_configurations(
            path="./figures/dataset/normal_{}_configs".format(name))
        dataset = RobotSimDataset(robot, num_samples)
        dataset.histogram(
            path="./figures/dataset/normal_{}_histogram".format(name))

        # uniform
        dataset = RobotSimDataset(robot, num_samples_limited, normal=False)
        dataset.plot_configurations(
            path="./figures/dataset/uniform_{}_configs".format(name))
        dataset = RobotSimDataset(robot, num_samples, normal=False)
        dataset.histogram(
            path="./figures/dataset/uniform_{}_histogram".format(name))
