import torch
from torch.utils.data import DataLoader
from models.CVAE import *
from robotsim_dataset import RobotSimDataset
import robotsim
from losses import VAE_loss_ROBOT_SIM
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import yaml

if __name__ == '__main__':

    '''
    Visualizes the performance of the cVAE
    For 1 sample TCP coordinate, samples from the latent space are drawn and and the joint angles of the robot are
    reconstructed and visualised.
    '''

    ####################################################################################################################
    # LOAD CONFIG
    ####################################################################################################################

    config = load_config('robotsim_cVAE.yaml', 'configs/')

    X_dim = config['input_dim']
    hidden_dim = config['hidden_dim']
    latent_dim = config['latent_dim']
    num_cond = config['condition_dim']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['lr_rate']
    weight_decay = config['weight_decay']
    variational_beta = config['variational_beta']
    use_gpu = config['use_gpu']
    PATH = config['weight_dir']
    dof = config['dof']

    ####################################################################################################################
    # LOAD DATASET
    ####################################################################################################################

    if dof == '2DOF':
        robot = robotsim.Robot2D2DoF([3, 2])
        # INPUT: 2 joint angles
        # OUTPUT: (x,y) coordinate of end-effector
        dataset = RobotSimDataset(robot, 1000)
    elif dof == '3DOF':
        robot = robotsim.Robot2D3DoF([3, 3, 3])
        # INPUT: 3 joint angles
        # OUTPUT: (x,y) coordinate of end-effector
        dataset = RobotSimDataset(robot, 100)
    else:
        raise Exception('Number of degrees of freedom ot supported')

    # train test split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [700000, 300000])

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    ####################################################################################################################
    # BUILD MODEL
    ####################################################################################################################

    cvae = CVAE(X_dim, hidden_dim, latent_dim, num_cond, classification=False)

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    cvae = cvae.to(device)

    cvae.load_weights(PATH)
    # epoch, loss = cvae.load_checkpoint(PATH=config['checkpoint_dir'] + 'CVAE_2DOF_epoch_10')

    # set to evaluation mode
    cvae.eval()

    ####################################################################################################################
    # TEST MODEL
    ####################################################################################################################

    test_loss_avg = []

    test_loss_avg.append(0)
    num_batches = 0

    print('Testing ...')
    for joint_batch, coord_batch in test_dataloader:
        joint_batch = joint_batch.to(device)

        # forward pass only accepts float
        joint_batch = joint_batch.float()
        coord_batch = coord_batch.float()

        # apply sine and cosine to joint angles
        joint_batch = preprocess(joint_batch)

        # forward propagation
        with torch.no_grad():
            image_batch_recon, latent_mu, latent_logvar = cvae(joint_batch, coord_batch)
            loss = VAE_loss_ROBOT_SIM(image_batch_recon, joint_batch, latent_mu, latent_logvar, variational_beta)

        test_loss_avg[-1] += loss.item()
        num_batches += 1

    test_loss_avg[-1] /= num_batches
    print('Average reconstruction error: %f' % (test_loss_avg[-1]))

    ####################################################################################################################
    # VISUALISATION
    ####################################################################################################################

    print('---------------SAMPLE GENERATION---------------')

    input = None
    if dof == '2DOF':
        # Specify initial joint angles
        # input = torch.Tensor([[-np.pi / 4, np.pi / 2]])
        input = torch.Tensor([[-np.pi / 3, np.pi / 3]])
    elif dof == '3DOF':
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
    robot.plot(joint_states=input.numpy(), path='figures/gt_configurations_' + str(dof) + '.png', separate_plots=False)

    # Generate joints angles from predefined tcp coordinates
    _x = []
    _y = []
    preds_joints = []

    for i in range(config['num_samples_config']):
        # create a random latent vector
        z = torch.randn(1, latent_dim).to(device)
        with torch.no_grad():
            recons_joint_angles = cvae.predict(z, tcp)
        preds = postprocess(recons_joint_angles)
        preds_joints.append(preds.numpy().tolist()[0])

        # Compute tcp coordinates resulting by performing forward kinematics on the generated joint angles
        coord = robot.forward(joint_states=preds.numpy().tolist()[0])
        _x.append(coord[0])
        _y.append(coord[1])

    # Plot generated configurations
    preds_joints = np.array(preds_joints)
    robot.plot(joint_states=preds_joints, path='figures/generated_configurations_' + str(dof) + '.png', separate_plots=False)

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
        z = cvae.visualise_z(input, tcp)

    fig = plt.figure()
    plt.title('Latent space')
    plt.xlabel('Z1')
    plt.ylabel('Z2')
    plt.scatter(z[:, 0], z[:, 1], c='g')
    plt.savefig('figures/Latent_space_' + str(dof) + '.png')

    print('-----------------------------------------------')