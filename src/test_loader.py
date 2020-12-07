from utils import *
from losses import *

def test_CVAE(model, config, dataloader, device):

    test_loss_avg = []

    test_loss_avg.append(0)
    num_batches = 0

    for joint_batch, coord_batch in dataloader:
        joint_batch = joint_batch.to(device)

        # forward pass only accepts float
        joint_batch = joint_batch.float()
        coord_batch = coord_batch.float()

        # apply sine and cosine to joint angles
        joint_batch = preprocess(joint_batch)

        # forward propagation
        with torch.no_grad():
            image_batch_recon, latent_mu, latent_logvar = model(joint_batch, coord_batch)
            loss = VAE_loss_ROBOT_SIM(image_batch_recon, joint_batch, latent_mu, latent_logvar, config['variational_beta'])

        test_loss_avg[-1] += loss.item()
        num_batches += 1

    test_loss_avg[-1] /= num_batches
    print('Average reconstruction error: %f' % (test_loss_avg[-1]))

def test_INN(model, config, dataloader, device):

    test_loss_avg = []

    test_loss_avg.append(0)
    num_batches = 0

    y_noise_scale = 1e-1
    zeros_noise_scale = 5e-2

    for x, y in dataloader:

        x, y = x.to(device), y.to(device)

        # forward pass only accepts float
        x = x.float()
        y = y.float()

        # apply sine and cosine to joint angles
        x = preprocess(x)

        # Insert noise
        # Padding in case xdim < total dim or yz_dim < total_dim
        pad_x = zeros_noise_scale * torch.randn(config['batch_size'], config['total_dim'] -
                                                config['input_dim'], device=device)

        pad_yz = zeros_noise_scale * torch.randn(config['batch_size'], config['total_dim'] -
                                                 config['output_dim'] - config['latent_dim'], device=device)

        y += y_noise_scale * torch.randn(config['batch_size'], config['output_dim'], dtype=torch.float,
                                         device=device)

        # Sample z from standard normal distribution
        z = torch.randn(config['batch_size'], config['latent_dim'], device=device)

        # Concatenate
        x = torch.cat((x, pad_x), dim=1)
        y = torch.cat((z, pad_yz, y), dim=1)

        # forward propagation
        with torch.no_grad():
            # forward propagation
            output = model(x)

            # shorten y and output for latent loss computation
            y_short = torch.cat((y[:, :config['latent_dim']], y[:, -config['output_dim']:]), dim=1)
            output_short = torch.cat((output[:, :config['latent_dim']], output[:, -config['output_dim']:].data), dim=1)

            L_y = MSELoss(output[:, config['latent_dim']:], y[:, config['latent_dim']:])

            L_z = MMD_loss(output_short, y_short, device)

            loss_forward = config['weight_Ly'] * L_y + config['weight_Lz'] * L_z

            loss = loss_forward.data.item()

        test_loss_avg[-1] += loss
        num_batches += 1

    test_loss_avg[-1] /= num_batches
    print('Average error: %f' % (test_loss_avg[-1]))