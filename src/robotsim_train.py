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

    '''
    Trains a conditional autoencoder on the robotsim dataset
    '''

    ####################################################################################################################
    # LOAD CONFIG
    ####################################################################################################################

    # config = load_config('robotsim_cVAE.yaml', 'configs/')
    config = load_config('robotsim_INN.yaml', 'configs/')

    model_name = 'INN'

    ####################################################################################################################
    # LOAD DATASET
    ####################################################################################################################

    if config['dof'] == '2DOF':
        robot = robotsim.Robot2D2DoF([3, 2])
        # INPUT: 2 joint angles
        # OUTPUT: (x,y) coordinate of end-effector
        dataset = RobotSimDataset(robot, 1000)
    elif config['dof'] == '3DOF':
        robot = robotsim.Robot2D3DoF([3, 3, 3])
        # INPUT: 3 joint angles
        # OUTPUT: (x,y) coordinate of end-effector
        dataset = RobotSimDataset(robot, 100)
    else:
        raise Exception('Number of degrees of freedom not supported')

    # train test split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [700000, 150000, 150000])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    ####################################################################################################################
    # BUILD MODEL
    ####################################################################################################################

    # model = CVAE(config['input_dim'], config['hidden_dim'], config['latent_dim'], config['condition_dim'],
    #              classification=False)

    model = INN(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'])

    device = torch.device("cuda:0" if config['use_gpu'] and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ####################################################################################################################
    # TRAINING
    ####################################################################################################################

    # train_loss_avg = train_CVAE(model=model, config=config, dataloader=train_dataloader, device=device)
    train_loss_avg = train_INN(model=model, config=config, dataloader=train_dataloader, device=device)

    model.save_weights(config['weight_dir'])

    fig = plt.figure()
    plt.title('AVG LOSS HISTORY')
    plt.xlabel('EPOCHS')
    plt.ylabel('AVG LOSS')
    plt.plot(train_loss_avg)
    plt.savefig('figures/avg_train_loss_' + model_name + '_' + str(config['dof']) + '.png')