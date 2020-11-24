import torch
from torch.utils.data import DataLoader
from models.CVAE import *
from robotsim_dataset import RobotSimDataset
import robotsim
from robotsim import RobotSim2D
from losses import VAE_loss_ROBOT_SIM

import matplotlib.pyplot as plt

if __name__ == '__main__':

    '''
    Visualizes the performance of the cVAE
    For 1 sample TCP coordinate, samples from the latent space are drawn and and the joint angles of the robot are
    reconstructed and visualised.
    '''

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    X_dim = 3 # number of robot links
    hidden_dim = 50 # number of neurons per fully connected layer
    latent_dim = 1
    num_cond = 2 # (x, y) coordinates of end-effector
    batch_size = 250
    variational_beta = 1 / 15
    use_gpu = False
    PATH = 'weights/ROBOTSIM_CVAE2'
    NUM_SAMPLES = 500 # number of samples which are drawn for z to generate joint angles

    ####################################################################################################################
    # LOAD DATASET
    ####################################################################################################################

    robot = robotsim.RobotSim2D(3, [3, 3, 3])

    # INPUT: 3 joint angles
    # OUTPUT: (x,y) coordinate of end-effector
    dataset = RobotSimDataset(robot, 100)

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

    '''
    >> INPUT:  tensor([[1.5867, 6.2832, 5.2677]])
    >> TCP:  tensor([[2.4286, 7.6212]])
    '''

    input = torch.Tensor([test_dataset.__getitem__(5)[0]])
    tcp = torch.Tensor([test_dataset.__getitem__(5)[1]])

    print('INPUT: ', input)
    print('TCP: ', tcp[0])

    robot = robotsim.RobotSim2D(3, [3, 3, 3])
    # robot.plot_configurations(joint_states=input.numpy())

    _x = []
    _y = []
    for i in range(NUM_SAMPLES):

        # create a random latent vector
        z = torch.randn(1, latent_dim).to(device)
        # print('Generated latent variable z: ', z)
        z = torch.cat((z, tcp), dim=1)

        with torch.no_grad():
            recons_joint_angles = cvae.decoder(z)

        coord = robot.forward(joint_states=recons_joint_angles.numpy().tolist()[0])
        # print('TCP coordinates after forward kinematics of generated joint angles: ', coord)
        # points.append(recons_joint_angles.numpy().tolist()[0])
        _x.append(coord[0])
        _y.append(coord[1])


    # red: gt (x,y)
    # green: (x, y) of generated joint angles
    fig = plt.figure()
    plt.title('TCP coordinates of generated joint angles')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(_x, _y, c='g')
    plt.scatter(tcp[0][0], tcp[0][1], c='r')
    plt.savefig('coord2.png')

    print('-----------------------------------------------')