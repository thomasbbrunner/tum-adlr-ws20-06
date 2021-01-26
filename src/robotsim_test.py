import torch
from torch.utils.data import DataLoader
from models.CVAE import *
from models.INN import *
from robotsim_dataset import RobotSimDataset
import robotsim
from losses import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import yaml
import json
import argparse

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

    # TODO: Make implementation compatible with GPU

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    N = 1
    M = 100
    percentile = 0.97

    ####################################################################################################################
    # LOAD CONFIG AND DATASET, BUILD MODEL
    ####################################################################################################################

    parser = argparse.ArgumentParser(
        description="Testing of neural networks for inverse kinematics.")
    parser.add_argument(
        "config_file", help="file containing configurations.")
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
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # load pre-trained weights
    model.load_weights(config['weight_dir'])

    # load pre-trained weights from checkpoint
    # epoch, loss = model.load_checkpoint(PATH=config['checkpoint_dir'] + model_name + '_' + config['dof'] + '_epoch_40')

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

        joint_states = robot.rejection_sampling(
            tcp_coordinates=N_y, num_samples=M, eps=0.05, mean=0, std=0.5)

        # generate plots for visualization for first sample
        if n == 0:

            print('N_x: ', N_x)
            print('N_y: ', N_y)

            # plot sample configuration from estimated posterior by rejection sampling
            robotsim.heatmap(
                joint_states, robot, highlight=22, transparency=0.2, 
                path='figures/rejection_sampling_{}_{}DOF.png'.format(config['model'], config['dof']))

            # Plot contour lines enclose the region containing 97% of the end points
            resimulation_tcp = robot.forward(joint_states=joint_states)
            resimulation_xy = resimulation_tcp[:, :2]
            plot_contour_lines(
                points=resimulation_xy, gt=N_y,
                PATH='figures/q_quantile_rejection_sampling_{}_{}DOF.png'.format(config['model'], config['dof']),
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
            robotsim.heatmap(
                pred_joint_states, robot, highlight=22, transparency=0.2,
                path="figures/predicted_posterior_{}_{}DOF.png".format(config['model'], config['dof']))

            # Plot contour lines enclose the region containing 97% of the end points
            resimulation_tcp = robot.forward(joint_states=pred_joint_states)
            resimulation_xy = resimulation_tcp[:, :2]
            plot_contour_lines(
                points=resimulation_xy, gt=N_y,
                PATH="figures/q_quantile_prediction_{}_{}DOF.png".format(config['model'], config['dof']),
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
    list_results.append(config['model'])
    list_results.append('{}DOF'.format(config['dof']))
    list_results.append('# trainable parameters: ' + str(num_trainable_parameters))
    list_results.append('N = ' + str(N))
    list_results.append('M = ' + str(M))
    list_results.append('Average error of posterior: ' + str(mismatch_avg[-1]))
    list_results.append('Average re-simulation error: ' + str(error_resim_avg[-1]))
    list_results.append('config: ' + str(config))

    with open('results_{}_{}DOF.json'.format(config['model'], config['dof']), 'w') as fout:
        for item in list_results:
            json.dump(item, fout)
            fout.write('\n')
