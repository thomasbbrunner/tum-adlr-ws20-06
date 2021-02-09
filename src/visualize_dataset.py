import torch
from torch.utils.data import DataLoader
import argparse

from train_loader import *
from models import *

if __name__ == '__main__':
    """
    Creating the datasets based on a Gaussian distribution with varying stddev
    --> figures are stored in figures/evaluation/dataset/
    """

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    DATASET_SAMPLES = 1e4
    NORMAL = True
    DOF = 20
    LINKS = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
    name = "20dof"

    ####################################################################################################################

    robot = robotsim.Planar(DOF, LINKS)
    dataset = RobotSimDataset(robot, DATASET_SAMPLES, normal=NORMAL)

    dataset.plot_configurations(
        path="./figures/evaluation/dataset/_{}_configs".format(name))
    dataset = RobotSimDataset(robot, DATASET_SAMPLES)
    dataset.histogram(
        path="./figures/evaluation/dataset/_{}_histogram".format(name))
