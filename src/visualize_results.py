import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    Creating the graphs for the comparison between both models with increasing DoFs
    --> evaluation graphs are stored in figures/evaluation/
    """

    # alternative figures

    s_x = 1e-7
    s_y = 1

    # INN: # of parameters
    params_INN = [1108872, 2772084, 997884, 2494380]

    for i in range(len(params_INN)):
        params_INN[i] = params_INN[i] * 1e-6

    # INN: e_posterior
    e_posterior_INN = [0.014587438760325313, 0.012290126254782081, 0.01165938847279176, 0.01248140924004838]
    e_resim_INN = [0.0023, 0.0041, 0.0035, 0.0038]


    # CVAE: # of parameters
    params_CVAE = [217128, 166814, 1022020, 1303226]
    for i in range(len(params_CVAE)):
        params_CVAE[i] = params_CVAE[i] * 1e-6

    # CVAE: e_posterior
    e_posterior_CVAE = [0.014851279410067945, 0.008709543236764148, 0.007258017353480682, 0.00765587985701859]
    e_resim_CVAE = [7.6100e-05, 0.0002, 0.0007, 0.0012]

    for i in range(len(e_posterior_INN)):
        e_posterior_INN[i] = e_posterior_INN[i] * 100
        e_resim_INN[i] = e_resim_INN[i] * 1000
        e_posterior_CVAE[i] = e_posterior_CVAE[i] * 100
        e_resim_CVAE[i] = e_resim_CVAE[i] * 1000

    # annotations
    annotations_cVAE = ['4DoF', '6DOF', '8DoF', '10DoF']
    annotations_INN = ['4DoF', '6DOF', '8DoF', '10DoF']

    ####################################################################################################################

    # alternative plot
    # e_posterior
    plt.figure()#figsize=(20, 15))

    plt.scatter(params_INN, e_posterior_INN, color="#073642", label="INN", marker='^')#, s=600)
    plt.scatter(params_CVAE, e_posterior_CVAE, color="#bc5090", label="CVAE", marker='*')#, s=600)

    for i, txt in enumerate(annotations_cVAE):
        # add space to void overlap with icons
        plt.annotate(" " + txt, (params_CVAE[i], e_posterior_CVAE[i]))#, fontsize=30)

    for i, txt in enumerate(annotations_INN):
        plt.annotate(" " + txt, (params_INN[i], e_posterior_INN[i]))#, fontsize=30)

    plt.grid(True, color="#93a1a1", alpha=0.9)
    # plt.xticks(fontsize=30, rotation=0)
    # plt.yticks(fontsize=30, rotation=0)
    plt.xlabel('number of parameters (1e6)')#, fontsize=30)
    plt.ylabel('error of posterior (1e-2)')#, fontsize=30)

    plt.legend()#frameon=False, fontsize=30)
    plt.title('Comparison between INN and cVAE')#, fontsize=40)
    plt.savefig('figures/evaluation/comparison_e_posterior_alternative.png', dpi=300)

    ####################################################################################################################

    # e_resim
    plt.figure()#figsize=(20, 15))

    plt.scatter(params_INN, e_resim_INN, color="#073642", label="INN", marker='^')#, s=600)
    plt.scatter(params_CVAE, e_resim_CVAE, color="#bc5090", label="CVAE", marker='*')#, s=600)

    for i, txt in enumerate(annotations_cVAE):
        # add space to void overlap with icons
        plt.annotate(" " + txt, (params_CVAE[i], e_resim_CVAE[i]))#, fontsize=30)

    for i, txt in enumerate(annotations_INN):
        plt.annotate(" " + txt, (params_INN[i], e_resim_INN[i]))#, fontsize=30)

    plt.grid(True, color="#93a1a1", alpha=0.9)
    # plt.xticks(fontsize=30, rotation=0)
    # plt.yticks(fontsize=30, rotation=0)
    plt.xlabel('number of parameters (1e6)')#, fontsize=30)
    plt.ylabel('re-simulation error (1e-3)')#, fontsize=30)
    plt.legend()#frameon=False, fontsize=30)
    plt.title('Comparison between INN and cVAE')#, fontsize=40)
    plt.savefig('figures/evaluation/comparison_e_resim_alternative.png', dpi=300)
