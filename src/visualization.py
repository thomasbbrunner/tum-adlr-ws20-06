import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    Creating the graphs for the comparison between both models with increasing DoFs
    --> evaluation graphs are stored in figures/evaluation/
    """

    ####################################################################################################################

    data_INN_post = {'2DoF': 0.061, '3DoF': 0.066, '4DoF': 0.0438, '6DoF': 0.173, '10DoF': 0.127, '15DoF': 0.072,
                     '25DoF': 0.127}
    names_INN_post = list(data_INN_post.keys())
    values_INN_post = list(data_INN_post.values())

    data_INN_resim = {'2DoF': 0.012, '3DoF': 0.035, '4DoF': 0.043, '6DoF': 0.991, '10DoF': 2.304, '15DoF': 7.067,
                      '25DoF': 16.988}
    names_INN_resim = list(data_INN_resim.keys())
    values_INN_resim = list(data_INN_resim.values())

    data_CVAE_post = {'2DoF': 0.077, '3DoF': 0.045, '4DoF': 0.062, '6DoF': 0.130, '10DoF': 0.400, '15DoF': 0.159,
                      '25DoF': 0.149}
    names_CVAE_post = list(data_CVAE_post.keys())
    values_CVAE_post = list(data_CVAE_post.values())

    data_CVAE_resim = {'2DoF': 0.003, '3DoF': 0.045, '4DoF': 0.005, '6DoF': 1.953, '10DoF': 5.250, '15DoF': 7.149,
                       '25DoF': 8.745}
    names_CVAE_resim = list(data_CVAE_resim.keys())
    values_CVAE_resim = list(data_CVAE_resim.values())

    ####################################################################################################################

    # alternative figures

    # INN: # of parameters
    # dummy values for now
    params_INN = [6876180]

    # INN: e_posterior
    # dummy values for now
    e_posterior_INN = [0.049]

    # CVAE: # of parameters
    # dummy values for now
    params_CVAE = [1303226, 395291]

    # CVAE: e_posterior
    # dummy values for now
    e_posterior_CVAE = [0.053, 0.060]

    # annotations
    annotations_cVAE = ['10DoF', '15DoF']
    annotations_INN = ['10DoF']

    ####################################################################################################################

    # e_posterior
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(names_INN_post, values_INN_post, color="#073642", label="INN", marker='o')
    plt.plot(names_CVAE_post, values_CVAE_post, color="#bc5090", label="CVAE", marker='o')
    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.ylabel('error of posterior')
    plt.legend(frameon=False)
    plt.title('Comparison between INN and cVAE')
    plt.savefig('figures/evaluation/comparison_e_posterior.jpg')

    # e_resim
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(names_INN_resim, values_INN_resim, color="#073642", label="INN", marker='o')
    plt.plot(names_CVAE_resim, values_CVAE_resim, color="#bc5090", label="CVAE", marker='o')
    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.ylabel('re-simulation error')
    plt.legend(frameon=False)
    plt.title('Comparison between INN and cVAE')
    plt.savefig('figures/evaluation/comparison_e_resim.jpg')

    # alternative plot
    # e_posterior
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(params_INN, e_posterior_INN, color="#073642", label="INN", marker='o')
    plt.plot(params_CVAE, e_posterior_CVAE, color="#bc5090", label="CVAE", marker='o')

    for i, txt in enumerate(annotations_cVAE):
        plt.annotate(txt, (params_CVAE[i], e_posterior_CVAE[i]))

    for i, txt in enumerate(annotations_INN):
        plt.annotate(txt, (params_INN[i], e_posterior_INN[i]))

    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.xlabel('number of parameters')
    plt.ylabel('error of posterior')
    plt.legend(frameon=False)
    plt.title('Comparison between INN and cVAE')
    plt.savefig('figures/evaluation/comparison_e_posterior_alternative.jpg')