import torch
from torch.utils.data import DataLoader
import json
import argparse

from models import *
from robotsim_dataset import RobotSimDataset
from losses import *
from utils import *

if __name__ == '__main__':
    """Evaluation of the respective model:
    _________________________________________________________________________________________
    
    Usage: call the script with a config file as an argument
    
    Examples:
    # name of file in configs/ directory
    python3 test.py robotsim_cVAE_4DOF.yaml
    
    # or direct path to a config file
    python3 test.py ./configs/robotsim_cVAE_4DOF.yaml
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
    
    """

    # TODO: Make implementation compatible with GPU

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    DATASET_SAMPLES = 1e6
    N = 100
    M = 100
    percentile = 0.97
    STD=0.2
    NORMAL = True

    ####################################################################################################################
    # LOAD CONFIG AND DATASET, BUILD MODEL
    ####################################################################################################################

    parser = argparse.ArgumentParser(
        description="Testing of neural networks for inverse kinematics.")
    parser.add_argument(
        "config_file", 
        help="File containing configurations. "
        "Can be a name of a file in the configs directory "
        "or the path to a config file.")
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

    dataset = RobotSimDataset(robot, DATASET_SAMPLES, normal=NORMAL, stddev=STD)

    TRAIN_SAMPLES = int(0.7 * DATASET_SAMPLES)
    TEST_SAMPLES = int(0.3 * DATASET_SAMPLES)
    # ensures that models are trained and tested on the same samples
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [TRAIN_SAMPLES, TEST_SAMPLES])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # load pre-trained weights
    # model.load_weights('{}weights_{}_{}DOF'.format(config['results_dir'], config['model'], config['dof']))
    # model.load_weights('./weights/weights_INN_6DOF')
    model.load_weights('./weights/results/ROBOTSIM_INN_6DOF')

    # load pre-trained weights from checkpoint
    # epoch, loss = model.load_checkpoint(
    #     PATH='{}checkpoint_{}_{}DOF_epoch_{}'
    #     .format(config['results_dir'], config['model'], config['dof'], 400))

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
                path='{}rejection_sampling_{}_{}DOF.png'.format(
                    config['results_dir'], config['model'], config['dof']))

            # Plot contour lines enclose the region containing 97% of the end points
            resimulation_tcp = robot.forward(joint_states=joint_states)
            resimulation_xy = resimulation_tcp[:, :2]
            plot_contour_lines(
                points=resimulation_xy, gt=N_y,
                PATH='{}q_quantile_rejection_sampling_{}_{}DOF.png'.format(
                    config['results_dir'], config['model'], config['dof']),
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
        pred_joint_states = postprocess(pred_joint_states[:, :config['input_dim']], config=config)

        # generate plots for visualization for first sample
        if n == 0:
            # plot sample configuration from predicted posterior
            robotsim.heatmap(
                pred_joint_states.cpu(), robot, highlight=22, transparency=0.2,
                path="{}predicted_posterior_{}_{}DOF.png".format(
                    config['results_dir'], config['model'], config['dof']))

            # Plot contour lines enclose the region containing 97% of the end points
            resimulation_tcp = robot.forward(joint_states=pred_joint_states.cpu())
            resimulation_xy = resimulation_tcp[:, :2]
            plot_contour_lines(
                points=resimulation_xy, gt=N_y,
                PATH="{}q_quantile_prediction_{}_{}DOF.png".format(
                    config['results_dir'], config['model'], config['dof']),
                percentile=percentile)

        # 3.
        # Calculate posterior mismatch between _p(x|y*) and p_gt(x|y*) with MMD

        error = MMD(gt_joint_states.to(device), pred_joint_states.to(device))
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
        _x = postprocess(_x, config=config)

        # perform forward kinemtatics on _x
        y_resim = torch.Tensor(robot.forward(joint_states=_x.cpu().detach()))
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

    with open('{}results_{}_{}DOF.json'.format(config['results_dir'], config['model'], config['dof']), 'w') as fout:
        for item in list_results:
            json.dump(item, fout)
            fout.write('\n')
