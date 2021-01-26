import torch

from torch.utils.data import DataLoader

from models.CVAE import *
from models.INN import *
from losses import VAE_loss_ROBOT_SIM

from robotsim_dataset import RobotSimDataset
import robotsim

import matplotlib.pyplot as plt
from utils import *
from train_loader import *
import argparse

'''
Training of the respective model:
_________________________________________________________________________________________

Usage: call the script with a config file as an argument

Examples:
# name of file in configs/ directory
python3 robotsim_train.py robotsim_cVAE_4DOF.yaml

# or direct path to a config file
python3 robotsim_train.py ./configs/robotsim_cVAE_4DOF.yaml
_________________________________________________________________________________________
'''

if __name__ == '__main__':

    ####################################################################################################################
    # LOAD CONFIG AND DATASET, BUILD MODEL
    ####################################################################################################################

    parser = argparse.ArgumentParser(
        description="Training of neural networks for inverse kinematics.")
    parser.add_argument(
        "config_file", 
        help="File containing configurations. "
        "Can be a name of a file in the configs directory "
        "or the path to a config file.")
    args = parser.parse_args()
    config = load_config(args.config_file)

    if config["model"] == "CVAE":
        model = CVAE(config)
    elif config["model"] == "INN":
        model = INN(config)
    else:
        raise ValueError(
            "Unknown model in config: {}".format(config["model"]))

    if config["robot"] in ("Planar", "planar"):
        robot = robotsim.Planar(config["dof"], config["len_links"])
    elif config["robot"] in ("Paper", "paper"):
        robot = robotsim.Paper(config["dof"], config["len_links"])
    else:
        raise ValueError(
            "Unknown robot in config: {}".format(config["robot"]))

    dataset = RobotSimDataset(robot, 1e6)

    # ensures that models are trained and tested on the same samples
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [700000, 300000])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ####################################################################################################################
    # TRAINING
    ####################################################################################################################

    print('Begin training ...')
    if config["model"] == "CVAE":
        train_CVAE(model=model, config=config, dataloader=train_dataloader, device=device)
    elif config["model"] == "INN":
        train_INN(model=model, config=config, dataloader=train_dataloader, device=device)
    else:
        raise Exception('Model not supported')

    model.save_weights(config['weight_dir'])
