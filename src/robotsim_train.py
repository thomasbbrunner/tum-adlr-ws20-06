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

if __name__ == '__main__':

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    model_name = 'INN'
    robot_dof = '4DOF'

    ####################################################################################################################
    # CHECK FOR VALID INPUT
    ####################################################################################################################

    if not (model_name == 'CVAE' or model_name == 'INN'):
        raise Exception('Model not supported')

    if not (robot_dof == '2DOF' or robot_dof == '3DOF' or robot_dof == '4DOF'):
        raise Exception('DOF not supported')

    ####################################################################################################################
    # LOAD CONFIG AND DATASET, BUILD MODEL
    ####################################################################################################################

    if model_name == 'CVAE':
        if robot_dof == '2DOF':
            config = load_config('robotsim_cVAE_2DOF.yaml', 'configs/')
            robot = robotsim.Robot2D2DoF([3, 2])
            dataset = RobotSimDataset(robot, 1e6)
        elif robot_dof == '3DOF':
            config = load_config('robotsim_cVAE_3DOF.yaml', 'configs/')
            robot = robotsim.Robot2D3DoF([0.5, 0.5, 1.0])
            dataset = RobotSimDataset(robot, 1e6)
        else:
            config = load_config('robotsim_cVAE_4DOF.yaml', 'configs/')
            robot = robotsim.Robot2D4DoF([0.5, 0.5, 0.5, 1.0])
            dataset = RobotSimDataset(robot, 1e6)
        model = CVAE(config)
    else:
        if robot_dof == '2DOF':
            config = load_config('robotsim_INN_2DOF.yaml', 'configs/')
            robot = robotsim.Robot2D2DoF([3, 2])
            dataset = RobotSimDataset(robot, 1e6)
        elif robot_dof == '3DOF':
            config = load_config('robotsim_INN_3DOF.yaml', 'configs/')
            robot = robotsim.Robot2D3DoF([0.5, 0.5, 1.0])
            dataset = RobotSimDataset(robot, 1e6)
        else:
            config = load_config('robotsim_INN_4DOF.yaml', 'configs/')
            robot = robotsim.Robot2D4DoF([0.5, 0.5, 0.5, 1.0])
            dataset = RobotSimDataset(robot, 1e6)
        model = INN(config)

    # ensures that models are trained and tested on the same samples
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [7000, 3000])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ####################################################################################################################
    # TRAINING
    ####################################################################################################################

    print('Begin training ...')
    if model_name == 'CVAE':
        train_CVAE(model=model, config=config, dataloader=train_dataloader, device=device)
    elif model_name == 'INN':
        train_INN(model=model, config=config, dataloader=train_dataloader, device=device)
    else:
        raise Exception('Model not supported')

    model.save_weights(config['weight_dir'])