import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data_INN_post = {'2DoF': 0.061, '3DoF': 0.066, '4DoF': 0.0438, '6DoF': 0.189, '10DoF': 0.116, '15DoF': 0.074}
    names_INN_post = list(data_INN_post.keys())
    values_INN_post = list(data_INN_post.values())

    data_INN_resim = {'2DoF': 0.012, '3DoF': 0.035, '4DoF': 0.043, '6DoF': 0.940, '10DoF': 2.410, '15DoF': 4.860}
    names_INN_resim = list(data_INN_resim.keys())
    values_INN_resim = list(data_INN_resim.values())

    data_CVAE_post = {'2DoF': 0.077, '3DoF': 0.045, '4DoF': 0.062, '6DoF': 0.156, '10DoF': 0.393, '15DoF': 0.173}
    names_CVAE_post = list(data_CVAE_post.keys())
    values_CVAE_post = list(data_CVAE_post.values())

    data_CVAE_resim = {'2DoF': 0.003, '3DoF': 0.045, '4DoF': 0.005, '6DoF': 1.957, '10DoF': 5.257, '15DoF': 7.044}
    names_CVAE_resim = list(data_CVAE_resim.keys())
    values_CVAE_resim = list(data_CVAE_resim.values())

    # fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # e_posterior
    plt.figure(figsize=(8, 5))
    plt.plot(names_INN_post, values_INN_post, color="#073642", label="INN", marker='*')
    plt.plot(names_CVAE_post, values_CVAE_post, color="r", label="CVAE", marker='*')
    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.ylabel('e_posterior')
    plt.legend(frameon=False)
    plt.title('Comparison between INN and cVAE')
    plt.savefig('comparison_e_posterior.jpg')

    # e_resim
    plt.figure(figsize=(8, 5))
    plt.plot(names_INN_resim, values_INN_resim, color="#073642", label="INN", marker='*')
    plt.plot(names_CVAE_resim, values_CVAE_resim, color="r", label="CVAE", marker='*')
    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.ylabel('e_resim')
    plt.legend(frameon=False)
    plt.title('Comparison between INN and cVAE')
    plt.savefig('comparison_e_resim.jpg')