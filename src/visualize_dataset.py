import torch
from torch.utils.data import DataLoader
import argparse

from train_loader import *
from models import *

if __name__ == '__main__':
    """
    Creating the datasets based on a Gaussian distribution with varying stddev
    --> figures are stored in figures/evaluation/dataset/
    --> visualizing the workspace which is used for the training and testing dataset
    """

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    DATASET_SAMPLES = 1e6
    NORMAL = True
    STD = [0.2] # [0.3, 0.5, 1.5] # [0.3] # [0.1, 0.25, 0.5, 0.8, 1.5]
    DOF = [6, 10, 15] # [25] # [6, 10, 15, 25]

    ####################################################################################################################

    # create  directory if it does not exist
    pathlib.Path("figures/evaluation/dataset/").mkdir(exist_ok=True)

    for dof in DOF:
        print("dof =", dof)
        links = [0.5 for i in range(dof-1)]
        links.append(1.0)
        robot = robotsim.Planar(dof, links)
        for std in STD:
            print("std =", std)
            dataset = RobotSimDataset(robot, DATASET_SAMPLES, normal=NORMAL, stddev=std)
            dataset.plot_configurations(
                path="figures/evaluation/dataset/normal_std_" + str(std) + "_" + str(dof) + "DoF.jpg")
            dataset = RobotSimDataset(robot, DATASET_SAMPLES, normal=NORMAL, stddev=std)
            dataset.histogram(
                path="figures/evaluation/dataset/normal_std_" + str(std) + "_" + str(dof) + "DoF_histogram.jpg")