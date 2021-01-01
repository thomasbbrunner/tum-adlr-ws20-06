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

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    model_name = 'INN'
    robot_dof = '3DOF'

    percentile = 0.97

    ####################################################################################################################
    # CHECK FOR VALID INPUT
    ####################################################################################################################

    if not (model_name == 'CVAE' or model_name == 'INN'):
        raise Exception('Model not supported')

    if not (robot_dof == '2DOF' or robot_dof == '3DOF'):
        raise Exception('DOF not supported')

    ####################################################################################################################
    # LOAD CONFIG AND DATASET, BUILD MODEL
    ####################################################################################################################

    if model_name == 'CVAE':
        if robot_dof == '2DOF':
            config = load_config('robotsim_cVAE_2DOF.yaml', 'configs/')
            robot = robotsim.Robot2D2DoF([3, 2])
            dataset = RobotSimDataset(robot, 1e4)
        else:
            config = load_config('robotsim_cVAE_3DOF.yaml', 'configs/')
            robot = robotsim.Robot2D3DoF([3, 2, 3])
            dataset = RobotSimDataset(robot, 1e4)
        model = CVAE(config)
    else:
        if robot_dof == '2DOF':
            config = load_config('robotsim_INN_2DOF.yaml', 'configs/')
            robot = robotsim.Robot2D2DoF([3, 2])
            dataset = RobotSimDataset(robot, 1e4)
        else:
            config = load_config('robotsim_INN_3DOF.yaml', 'configs/')
            robot = robotsim.Robot2D3DoF([3, 2, 3])
            dataset = RobotSimDataset(robot, 1e4)
        model = INN(config)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [7000, 3000])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # load pre-trained weights
    model.load_weights(config['weight_dir'])

    # load pre-trained weights from checkpoint
    # epoch, loss = model.load_checkpoint(PATH=config['checkpoint_dir'] + model_name + '_' + config['dof'] + '_epoch_20')

    # set to evaluation mode
    model.eval()

    ####################################################################################################################
    # MODEL EVALUATION
    ####################################################################################################################

    # Compute RMSE
    test_rsme_avg = []
    test_rsme_avg.append(0)
    num_batches = 0

    for joint_batch, tcp_batch in test_dataloader:

        joint_batch = joint_batch.to(device)
        tcp_batch = tcp_batch.to(device)

        # forward pass only accepts float
        joint_batch = joint_batch.float()
        tcp_batch = tcp_batch.float()

        _x = model.predict(tcp_batch, device)
        rmse = RMSE(_x, joint_batch)

        test_rsme_avg[-1] += rmse.detach()
        num_batches += 1

    test_rsme_avg[-1] /= num_batches
    print('Average RMSE between gt joints and generated joints: %f' % (test_rsme_avg[-1]))

    ####################################################################################################################
    # VISUALISATION
    ####################################################################################################################

    # input = None
    # if config['dof'] == '2DOF':
    #     input = torch.Tensor([[-np.pi / 6, np.pi / 6]])
    # else:
    #     input = torch.Tensor([[-np.pi / 8, np.pi / 8, -np.pi / 8]])
    #
    # # compute resulting tcp coordinates
    # tcp = robot.forward(joint_states=input.detach())
    # tcp = torch.Tensor([[tcp[0], tcp[1]]])
    # # print('tcp coordinates: ', tcp)
    #
    # # Plot ground truth configuration
    # robot.plot(joint_states=input.detach(), path='figures/gt_configurations_' + str(config['dof']) + '.png',
    #            separate_plots=False)

    tcp = torch.Tensor([[8.0, 0.0]])
    # tcp = torch.Tensor([[5.0, -6.0]])

    # Generate joints angles from predefined tcp coordinates
    _x, _y, preds_joints = [], [], []

    for i in range(config['num_samples_config']):
        preds = model.predict(tcp, device)
        preds_joints.append(preds.detach().tolist()[0])

    # Plot generated configurations
    preds_joints = np.array(preds_joints)
    robot.plot(joint_states=preds_joints, path='figures/generated_configurations_' + model_name + '_' +
                                               str(config['dof']) + '.png', separate_plots=False)

    # Plot contour lines enclose the region conaining 97% of the end points
    resimulation_tcp = robot.forward(joint_states=preds_joints)
    resimulation_xy = resimulation_tcp[:, :2]

    tcp_squeezed = torch.squeeze(tcp)

    plot_contour_lines(config, resimulation_xy, gt=tcp_squeezed.numpy(), percentile=percentile)

    ####################################################################################################################

    # if model_name == 'CVAE':
    #     for i in range(config['num_samples_config']):
    #         pred_joint_angles = model.predict(tcp, device)
    #         # print('pred_joint_angles: ', pred_joint_angles)
    #         preds = postprocess(pred_joint_angles)
    #         preds_joints.append(preds.numpy().tolist()[0])
    # else:
    #     invalid_preds = 0
    #     for i in range(config['num_samples_config']):
    #         pred_joint_angles = model.predict(tcp, device)
    #         if torch.any(pred_joint_angles < -1.0) or torch.any(pred_joint_angles > 1.0):
    #             invalid_preds = invalid_preds + 1
    #             continue
    #         # print('pred_joint_angles: ', pred_joint_angles)
    #         preds = postprocess(pred_joint_angles)
    #         preds_joints.append(preds.numpy().tolist()[0])
    #
    #     print('INVALID PREDICTIONS / TOTAL PREDICTIONS: %i / %i' % (invalid_preds, config['num_samples_config']))

    # if model_name == 'CVAE':
    #     # visualise latent space
    #     input = []
    #     tcp = []
    #     for i in range(config['num_samples_latent']):
    #         input.append(test_dataset.__getitem__(i)[0])
    #         tcp.append(test_dataset.__getitem__(i)[1])
    #
    #     input = torch.Tensor(input)
    #     tcp = torch.Tensor(tcp)
    #
    #     # forward pass only accepts float
    #     input = input.float()
    #     tcp = tcp.float()
    #
    #     # apply sine and cosine to joint angles
    #     # input = preprocess(input)
    #
    #     # forward propagation
    #     with torch.no_grad():
    #         z = model.visualise_z(input, tcp)
    #
    #     print('size of z: ', z)
    #
    #     # fig = plt.figure()
    #     # plt.title('Latent space')
    #     # plt.xlabel('Z1')
    #     # plt.ylabel('Z2')
    #     # plt.scatter(z[:, 0], z[:, 1], c='g')
    #     # plt.savefig('figures/Latent_space_' + str(config['dof']) + '.png')
    #
    #     fig = plt.figure()
    #     plt.title('Latent space')
    #     plt.xlabel('Z')
    #     # plt.plot(z[:, 0], c='g')
    #     plt.plot(z[:, 0], np.zeros_like(z[:, 0]) + 0., 'x')
    #     plt.savefig('figures/Latent_space_CVAE_' + str(config['dof']) + '.png')
    #
    # else:
    #     # visualise latent space
    #     input = []
    #     for i in range(config['num_samples_latent']):
    #         input.append(test_dataset.__getitem__(i)[0])
    #
    #     input = torch.Tensor(input)
    #
    #     # forward pass only accepts float
    #     input = input.float()
    #
    #     # apply sine and cosine to joint angles
    #     # input = preprocess(input)
    #     model.visualise_z(config, input)
