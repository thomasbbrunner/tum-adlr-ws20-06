import torch
from torch.utils.data import DataLoader
from models.CVAE import *
from robotsim_dataset import RobotSimDataset
import robotsim
from robotsim import RobotSim2D
from losses import VAE_loss_ROBOT_SIM
from utils import *
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

    NUM_SAMPLES = config['num_samples']

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
        with torch.no_grad():
            recons_joint_angles = cvae.predict(z, tcp)
        preds = postprocess(recons_joint_angles)
        # print('PREDS OF JOINT ANGLES: ', preds)
        coord = robot.forward(joint_states=preds.numpy().tolist()[0])
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
    plt.savefig('TCP_coordinates.png')

    # visualise latent space
    input = []
    tcp = []
    for i in range(NUM_SAMPLES):
        input.append(test_dataset.__getitem__(i)[0])
        tcp.append(test_dataset.__getitem__(i)[1])

    input = torch.Tensor(input)
    tcp = torch.Tensor(tcp)

    # forward pass only accepts float
    input = input.float()
    tcp = tcp.float()

    # apply sine and cosine to joint angles
    input = preprocess(input)
    print(tcp)

    # forward propagation
    with torch.no_grad():
        z = cvae.visualise_z(input, tcp)

    # print(z)

    fig = plt.figure()
    plt.title('Latent space')
    plt.xlabel('Z1')
    plt.ylabel('Z2')
    plt.scatter(z[:, 0], z[:, 1], c='g')
    plt.savefig('Latent_space.png')

    print('-----------------------------------------------')