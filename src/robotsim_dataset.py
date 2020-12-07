
import matplotlib.pyplot as plt
import numpy as np
import pdb
from torch.utils.data import Dataset, DataLoader

import robotsim


class RobotSimDataset(Dataset):
    """
    TODO: 
        * check distribution of samples in cartesian space
            (uniform in angle space, but not in cartesian!)
            (plot in heatmap mode)
        * randomization
    """

    def __init__(self, robot, num_samples=100):
        """Dataset for robot simulation.

        Dataset is generated on demand. 
        Entire dataset is stored in memory.

        Generation of 1e6 samples takes <1s.
        and requires 48 MB of memory 
        (for 3 links).

        Total number of samples is num_samples^(num_dof).

        Args:
            robot: instance of RobotSim
            num_samples: number of evenly spaced samples for each joint.

        Example usage:
        # creation of dataset for planar 3DoF robot
        # with 100*100*100 samples
        >>> robot = robotsim.Robot2D3DoF([3, 3, 3])
        >>> dataset = RobotSimDataset(robot, 100)
        """

        joint_samples = robot.get_joint_samples(num_samples)

        # compute all possible permutations of the joint samples
        self.joint_states = np.array(
            np.meshgrid(*joint_samples)).T.reshape(-1, robot.NUM_DOF)

        self.tcp_coords = robot.forward(self.joint_states)

        if self.tcp_coords.shape[0] != self.joint_states.shape[0]:
            raise RuntimeError(
                "Inconsistent sizes in dataset contents.")

        self.len = self.tcp_coords.shape[0]

    def __len__(self):
        """Returns number of samples in dataset.
        """
        return self.len

    def __getitem__(self, item):
        """Returs tuple with joint states and TCP coordinates.
        """
        # exclude tcp orientation
        return self.joint_states[item], self.tcp_coords[item][:2]

    def plot(self, show=False):

        fig, ax = plt.subplots()
        ax.grid()
        ax.scatter(
            self.tcp_coords[:, 0],
            self.tcp_coords[:, 1],
            c='r', s=2)

        if show:
            plt.show()


if __name__ == "__main__":

    robot = robotsim.Robot2D2DoF([3, 2])
    dataset = RobotSimDataset(robot, 10)
    dataset.plot()

    robot = robotsim.Robot2D3DoF([3, 2, 3])
    dataset = RobotSimDataset(robot, 10)
    dataset.plot()

    robot = robotsim.Robot2D4DoF([3, 2, 3])
    dataset = RobotSimDataset(robot, 10)
    dataset.plot()

    plt.show()
