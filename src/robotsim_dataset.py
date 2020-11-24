
import numpy as np
from torch.utils.data import Dataset, DataLoader

import robotsim


class RobotSimDataset(Dataset):
    """
    TODO: randomization
    """

    def __init__(self, robot_sim, res=100):
        """Dataset for robot simulation.

        Dataset is generated on demand. 
        Entire dataset is stored in memory.

        Generation of 1e6 samples takes <1s.
        and requires 48 MB of memory 
        (for 3 links).

        Parameters:
        robot_sim: instance of RobotSim
        res: number of evenly spaced samples 
            for each joint state in range [0 to 2*pi]

        Example usage:
        # creation of dataset for planar 3DoF robot
        # with 100*100*100 samples
        >>> robot = robotsim.RobotSim2D(3, [3, 3, 3])
        >>> dataset = RobotSimDataset(robot, 100)
        """

        # uniformly spaced values in angle space
        angle_range = [
            np.linspace(0, 2*np.pi, res) for _ in range(0, robot_sim.num_links)]

        self.joint_values = np.array(
            np.meshgrid(*angle_range)).T.reshape(-1, robot_sim.num_links)

        self.tcp_coords = robot_sim.forward(self.joint_values)

        if self.tcp_coords.shape[0] != self.joint_values.shape[0]:
            raise RuntimeError(
                "Inconsistent sizes in dataset contents.")

        self.len = self.tcp_coords.shape[0]

        # visulization of dataset
        # robot_sim.plot_configurations(self.joint_values, separate_plots=False)

    def __len__(self):
        """Returns number of samples in dataset.
        """
        return self.len

    def __getitem__(self, item):
        """Returs tuple with joint states and TCP coordinates.
        """
        return self.joint_values[item], self.tcp_coords[item]
