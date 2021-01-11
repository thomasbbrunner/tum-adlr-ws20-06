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
import json

from test_loader import *

'''
Evaluation of the respective model:
_________________________________________________________________________________________

x*: ground truth input joint angles of the robot
y*: ground truth labels/ end-effector coordinated of the robot
N: # of test labels drawn from the test dataset
M: # of generated estimates for each test label to produce full posterior distribution

_________________________________________________________________________________________

1. Generate gt estimate p_gt(x|y*) obtained by rejection sampling
2. Predict posterior distribution _p(x|y*) with model
3. Calculate posterior mismatch between _p(x|y*) and p_gt(x|y*) with MMD
4. Apply the forward process f to the generated samples _p(x|y*): y_resim = f(_p(x|y*))
5. Measure the re-simulation error between y_resim and y*

'''

if __name__ == '__main__':

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    model_name = 'INN'
    robot_dof = '2DOF'

    N = 100
    M = 100
    percentile = 0.97

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
            robot = robotsim.Robot2D2DoF([0.5, 1])
            dataset = RobotSimDataset(robot, 1e6)
            dof=2
        elif robot_dof == '3DOF':
            config = load_config('robotsim_cVAE_3DOF.yaml', 'configs/')
            robot = robotsim.Robot2D3DoF([0.5, 0.5, 1.0])
            dataset = RobotSimDataset(robot, 1e6)
            dof = 3
        else:
            config = load_config('robotsim_cVAE_4DOF.yaml', 'configs/')
            robot = robotsim.Robot2D4DoF([0.5, 0.5, 0.5, 1.0])
            dataset = RobotSimDataset(robot, 1e6)
            dof = 4
        model = CVAE(config)
    else:
        if robot_dof == '2DOF':
            config = load_config('robotsim_INN_2DOF.yaml', 'configs/')
            robot = robotsim.Robot2D2DoF([0.5, 1])
            dataset = RobotSimDataset(robot, 1e6)
            dof = 2
        elif robot_dof == '3DOF':
            config = load_config('robotsim_INN_3DOF.yaml', 'configs/')
            robot = robotsim.Robot2D3DoF([0.5, 0.5, 1.0])
            dataset = RobotSimDataset(robot, 1e6)
            dof = 3
        else:
            config = load_config('robotsim_INN_4DOF.yaml', 'configs/')
            robot = robotsim.Robot2D4DoF([0.5, 0.5, 0.5, 1.0])
            dataset = RobotSimDataset(robot, 1e6)
            dof = 4
        model = INN(config)

    # ensures that models are trained and tested on the same samples
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [700000, 300000])
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

    # Average mismatch
    mismatch_avg = []
    mismatch_avg.append(0)
    for n in range(N):

        N_x = test_dataset.__getitem__(n)[0]
        N_y = test_dataset.__getitem__(n)[1]

        print('n / N : %i / %i' % (n, N))

        # 1.
        # Generate gt estimate p_gt(x|y*) obtained by rejection sampling with M samples

        joint_states = rejection_sampling(robot=robot, tcp=N_y, dof=dof, samples=M)

        # generate plots for visualization for first sample
        if n == 0:
            # plot sample configuration from estimated posterior by rejection sampling
            robot.plot(joint_states, path='figures/rejection_sampling_' + str(config['name']) + '_' + str(config['dof']) + '.png',
                       separate_plots=False)
            # Plot contour lines enclose the region containing 97% of the end points
            resimulation_tcp = robot.forward(joint_states=joint_states)
            resimulation_xy = resimulation_tcp[:, :2]
            plot_contour_lines(points=resimulation_xy, gt=N_y,
                               PATH='figures/q_quantile_rejection_sampling_' + config['name'] + '_' + config['dof'] + '.png',
                               percentile=percentile)

        gt_joint_states = torch.Tensor(joint_states)

        # 2.
        # Predict posterior distribution _p(x | y *) with model

        # Create M x 2 array as y input for model
        gen_tcp = np.zeros(shape=(M, 2))
        gen_tcp[:, 0] = N_y[0]
        gen_tcp[:, 1] = N_y[1]
        gen_tcp = torch.Tensor(gen_tcp)
        gen_tcp = gen_tcp.to(device)

        # predict posterior distribution based on M samples
        pred_joint_states = model.predict(tcp=gen_tcp, device=device)

        # Post-ptrocessing
        pred_joint_states = postprocess(pred_joint_states)

        # generate plots for visualization for first sample
        if n == 0:
            # plot sample configuration from predicted posterior
            robot.plot(pred_joint_states, path='figures/predicted_posterior_' + model_name + '_' +
                                               str(config['dof']) + '.png',
                       separate_plots=False)
            # Plot contour lines enclose the region containing 97% of the end points
            resimulation_tcp = robot.forward(joint_states=pred_joint_states)
            resimulation_xy = resimulation_tcp[:, :2]
            plot_contour_lines(points=resimulation_xy, gt=N_y,
                               PATH='figures/q_quantile_prediction_' + config['name'] + '_' + config[
                                   'dof'] + '.png',
                               percentile=percentile)

        # 3.
        # Calculate posterior mismatch between _p(x|y*) and p_gt(x|y*) with MMD

        error = MMD(gt_joint_states, pred_joint_states, device=device)
        mismatch_avg[-1] += error.item()

    # Average error over N different observations y*
    mismatch_avg[-1] /= N
    print('Average error of posterior: %.3f' % mismatch_avg[-1])

    # 4. and 5.
    # Apply the forward process f to the generated samples _p(x|y*): y_resim = f(_p(x|y*)) and
    # measure the re-simulation error between y_resim and y*

    # Now consider the whole test dataset
    num_batches = 0
    error_resim_avg = []
    error_resim_avg.append(0)
    for joint_batch, tcp_batch in test_dataloader:

        joint_batch = joint_batch.to(device)
        tcp_batch = tcp_batch.to(device)

        # only accepts float
        joint_batch = joint_batch.float()
        tcp_batch = tcp_batch.float()

        _x = model.predict(tcp_batch, device)

        # Post-ptrocessing
        _x = postprocess(_x)

        # perform forward kinemtatics on _x
        y_resim = torch.Tensor(robot.forward(joint_states=_x.detach()))
        y_resim = y_resim.to(device)

        # Exclude orientation
        y_resim = y_resim[:, :2]
        error_resim = MSE(y_resim, tcp_batch, reduction='mean')
        error_resim_avg[-1] += error_resim
        num_batches += 1

    error_resim_avg[-1] /= num_batches
    print('Average re-simulation error: %.3f' % error_resim_avg[-1])

    # show trainable parameters
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    num_trainable_parameters = sum(p.numel() for p in model.parameters())

    list_results = []
    list_results.append(config['name'])
    list_results.append(config['dof'])
    list_results.append('# trainable parameters: ' + str(num_trainable_parameters))
    list_results.append('N = ' + str(N))
    list_results.append('M = ' + str(M))
    list_results.append('Average error of posterior: ' + str(mismatch_avg[-1]))
    list_results.append('Average re-simulation error: ' + str(error_resim_avg[-1]))
    list_results.append('config: ' + str(config))

    with open('results_' + model_name + '_' + robot_dof + '.json', 'w') as fout:
        for item in list_results:
            json.dump(item, fout)
            fout.write('\n')