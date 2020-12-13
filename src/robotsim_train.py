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

    # model_name = 'CVAE'
    model_name = 'INN'
    robot_dof = '2DOF'

    ####################################################################################################################
    # LOAD DATASET
    ####################################################################################################################

    if model_name == 'CVAE':
        config = load_config('robotsim_cVAE.yaml', 'configs/')
    elif model_name == 'INN':
        if robot_dof == '2DOF':
            config = load_config('robotsim_INN_2DOF.yaml', 'configs/')
        elif robot_dof == '3DOF':
            config = load_config('robotsim_INN_3DOF.yaml', 'configs/')
        else:
            raise Exception('DOF not supported for this model')
    else:
        raise Exception('Model not supported')

    if config['dof'] == '2DOF':
        robot = robotsim.Robot2D2DoF([3, 2])
        # INPUT: 2 joint angles
        # OUTPUT: (x,y) coordinate of end-effector
        dataset = RobotSimDataset(robot, 1e6)
    elif config['dof'] == '3DOF':
        robot = robotsim.Robot2D3DoF([3, 3, 3])
        # INPUT: 3 joint angles
        # OUTPUT: (x,y) coordinate of end-effector
        dataset = RobotSimDataset(robot, 1e6)
    else:
        raise Exception('Number of degrees of freedom not supported')

    # train test split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [700000, 150000, 150000])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    ####################################################################################################################
    # BUILD MODEL
    ####################################################################################################################

    if model_name == 'CVAE':
        model = CVAE(config, classification=False)
    elif model_name == 'INN':
        model = INN(config)
    else:
        raise Exception('Model not supported')

    device = torch.device("cuda:0" if config['use_gpu'] and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # show the number of trainable parameters


    ####################################################################################################################
    # TRAINING
    ####################################################################################################################

    if model_name == 'CVAE':
        train_loss_avg = train_CVAE(model=model, config=config, dataloader=train_dataloader, device=device)
    elif model_name == 'INN':
        train_loss_avg = train_INN(model=model, config=config, dataloader=train_dataloader, device=device)
    else:
        raise Exception('Model not supported')

    model.save_weights(config['weight_dir'])

    fig = plt.figure()
    plt.title('AVG LOSS HISTORY')
    plt.xlabel('EPOCHS')
    plt.ylabel('AVG LOSS')
    plt.plot(train_loss_avg)
    plt.savefig('figures/avg_train_loss_' + model_name + '_' + str(config['dof']) + '.png')