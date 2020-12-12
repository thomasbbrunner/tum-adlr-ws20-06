import torch
from torch.utils.data import DataLoader
from models.CVAE import *
from models.INN import *
from robotsim_dataset import RobotSimDataset
import robotsim
from losses import VAE_loss_ROBOT_SIM
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import yaml

from test_loader import *

if __name__ == '__main__':

    '''
    Visualizes the performance of the model
    For 1 sample TCP coordinate, samples from the latent space are drawn and and the joint angles of the robot are
    reconstructed and visualized.
    '''

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    # model_name = 'CVAE'
    model_name = 'INN'

    ####################################################################################################################
    # LOAD DATASET
    ####################################################################################################################

    if model_name == 'CVAE':
        config = load_config('robotsim_cVAE.yaml', 'configs/')
    elif model_name == 'INN':
        config = load_config('robotsim_INN.yaml', 'configs/')
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
        raise Exception('Number of degrees of freedom ot supported')

    # train test split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [700000, 300000])

    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

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

    # model.load_weights(config['weight_dir'])
    epoch, loss = model.load_checkpoint(PATH=config['checkpoint_dir'] + model_name + '_' + config['dof'] + '_epoch_20')

    # set to evaluation mode
    model.eval()

    ####################################################################################################################
    # TEST MODEL
    ####################################################################################################################

    if model_name == 'CVAE':
        test_CVAE(model, config, test_dataloader, device)
    elif model_name == 'INN':
        test_INN(model, config, test_dataloader, device)
    else:
        raise Exception('Model not supported')

    ####################################################################################################################
    # VISUALISATION
    ####################################################################################################################

    print('---------------SAMPLE GENERATION---------------')

    input = None
    if config['dof'] == '2DOF':
        # Specify initial joint angles
        # input = torch.Tensor([[-np.pi / 4, np.pi / 2]])
        input = torch.Tensor([[-np.pi / 3, np.pi / 3]])
    elif config['dof'] == '3DOF':
        # Specify initial joint angles
        # input = torch.Tensor([[-np.pi / 4, np.pi / 2, -np.pi / 4]])
        input = torch.Tensor([[0, 0, 0]])
    else:
        raise Exception('Number of degrees of freedom ot supported')

    # compute resulting tcp coordinates
    tcp = robot.forward(joint_states=input.numpy())
    tcp_x = tcp[0]
    tcp_y = tcp[1]
    tcp = torch.Tensor([[tcp_x, tcp_y]])
    print('tcp coordinates: ', tcp)

    # Plot ground truth configuration
    robot.plot(joint_states=input.numpy(), path='figures/gt_configurations_' + str(config['dof']) + '.png', separate_plots=False)

    # Generate joints angles from predefined tcp coordinates
    _x = []
    _y = []
    preds_joints = []
    preds_joints_valid = []

    if model_name == 'CVAE':
        for i in range(config['num_samples_config']):
            pred_joint_angles = model.predict(tcp, device)
            # print('pred_joint_angles: ', pred_joint_angles)
            preds = postprocess(pred_joint_angles)
            preds_joints.append(preds.numpy().tolist()[0])
    else:
        for i in range(config['num_samples_config']):
            pred_joint_angles = model.predict(tcp, device)
            if torch.any(pred_joint_angles < -1.0) or torch.any(pred_joint_angles > 1.0):
                continue
            print('pred_joint_angles: ', pred_joint_angles)
            preds = postprocess(pred_joint_angles)
            preds_joints.append(preds.numpy().tolist()[0])

    # Plot generated configurations
    preds_joints = np.array(preds_joints)
    robot.plot(joint_states=preds_joints, path='figures/generated_configurations_' + model_name + '_' + str(config['dof']) + '.png', separate_plots=False)

    if model_name == 'CVAE':
        # visualise latent space
        input = []
        tcp = []
        for i in range(config['num_samples_latent']):
            input.append(test_dataset.__getitem__(i)[0])
            tcp.append(test_dataset.__getitem__(i)[1])

        input = torch.Tensor(input)
        tcp = torch.Tensor(tcp)

        # forward pass only accepts float
        input = input.float()
        tcp = tcp.float()

        # apply sine and cosine to joint angles
        input = preprocess(input)

        # forward propagation
        with torch.no_grad():
            z = model.visualise_z(input, tcp)

        fig = plt.figure()
        plt.title('Latent space')
        plt.xlabel('Z1')
        plt.ylabel('Z2')
        plt.scatter(z[:, 0], z[:, 1], c='g')
        plt.savefig('figures/Latent_space_' + str(config['dof']) + '.png')


    print('-----------------------------------------------')