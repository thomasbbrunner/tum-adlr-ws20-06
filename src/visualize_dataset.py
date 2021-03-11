import torch
from torch.utils.data import DataLoader
import argparse
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

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

    DATASET_SAMPLES = 1e4
    NORMAL = True
    STD = [0.2] # [0.3, 0.5, 1.5] # [0.3] # [0.1, 0.25, 0.5, 0.8, 1.5]
    DOF = [4, 6, 8, 10] # [25] # [6, 10, 15, 25]

    ####################################################################################################################

    # create  directory if it does not exist
    pathlib.Path("figures/evaluation/dataset/").mkdir(exist_ok=True)

    fig = Figure(figsize=(9,5))
    # fig.suptitle("Configurations in Dataset (subset of $10^4$ samples)")
    gridspec_kw = {"width_ratios": [3/18, 4/18, 5/18, 6/18]}
    axs = fig.subplots(nrows=1, ncols=4, gridspec_kw=gridspec_kw, sharey=True)

    for i, dof in enumerate(DOF):
        print("dof =", dof)
        links = [0.5 for i in range(dof-1)]
        links.append(1.0)
        robot = robotsim.Planar(dof, links)
        for std in STD:
            print("std =", std)
            dataset = RobotSimDataset(robot, DATASET_SAMPLES, normal=NORMAL, stddev=std)
            dataset.plot_configurations(
                ax=axs[i])
            
            axs[i].grid()

    axs[0].set_ylabel("$x_2$")
    axs[2].set_xlabel("$x_1$")
    axs[0].set_xlim([0, 3])
    axs[1].set_xlim([0, 4])
    axs[2].set_xlim([0, 5])
    axs[3].set_xlim([0, 6])
    fig.savefig("figures/evaluation/normal_std_" + str(std).replace(".", "_") + "_all_DoF.png", dpi=300)
