import torch

import numpy as np
import matplotlib.pyplot as plt

morph = False
if morph:
    u_true_1 = np.array([[0,1,0], [0,1,0]])
    # v_true_1 = np.array([[1,0,0], [1,0,0]])
    v_true_1 = np.array([[1,0], [1,0]])
    z_true_1 = np.array([[5,0], [-5,0]])

    u_true_2 = np.array([[-1.4815, 1.09, 0], [-2.9630, 1.1852, 0]])
    # v_true_2 = np.array([[-2.6698, 0.1809, 0], [-4.67, 0.2810, 0]])
    v_true_2 = np.array([[-2.6698, 0.1809], [-4.67, 0.2810]])
    z_true_2 = np.array([[5.9937, -0.0699], [5.9937, 0.0699]])

    u_true_3 = np.array([[1.4815, 0.9074, 0], [2.9630, 0.8148, 0]])
    # v_true_3 = np.array([[-2.6698, 0.1809, 0], [-4.67, 0.2810, 0]])
    v_true_3 = np.array([[-2.6698, 0.1809], [-4.67, 0.2810]])
    z_true_3 = np.array([[5.9937, -0.0699], [-5.9937, 0.0699]])

else:

    u_true_1 = np.array([[0, 1.25, .007], [0, 1, .005]])
    v_true_1 = np.array([[3, .1], [.1, .1]])
    g_poly = np.array([-1.1, -0.155, 0.0017])
    z_true_1 = np.array([g_poly, -g_poly])


    u_true_2 = np.array([[0, 1, .001], [0, 0.75, .001]])
    v_true_2 = np.array([[1, .25], [1, .1]])
    g_poly = np.array([0, -0.155, 0.0017])
    z_true_2 = np.array([g_poly, -g_poly])


    u_true_3 = torch.Tensor([[0, 1, -.003], [0, 1, .005]])
    v_true_3 = torch.Tensor([[1, .25], [5, 0.25]])
    g_poly = np.array([-1, -0.1, 0.0012])
    z_true_3 = torch.Tensor([g_poly, -g_poly])


ages = np.arange(1, 91)

poly_deg_2 = np.array([ages ** d for d in range(3)])
# poly_deg_1 = poly_deg_2
poly_deg_1 = np.array([ages ** d for d in range(2)])


datasets = [
    'Synth1',
    'Synth2',
    'Synth3',

    # 'AgeDB',
    # 'MORPH',
    # 'APPA-REAL',
    # 'IMDB',
    # 'UTKF',
    # 'LFW',
    # 'CPMRD',
    # 'INET',
    # 'GroupPhotos',
    # 'PAL',
    # 'PubFig',
    # 'SchoolClasses',    
]

plots_folder = 'results/plots'

splits = [
    # 'results/checkpoints/checkpoints_23.08_ADAM_LCM/1_checkpoint.pt',
    # 'results/checkpoints/checkpoints_23.08_ADAM_LCM/2_checkpoint.pt',
    # 'results/checkpoints/checkpoints_23.08_ADAM_LCM/3_checkpoint.pt',
    
    'results/checkpoints/checkpoints/1_checkpoint.pt',
    'results/checkpoints/checkpoints/2_checkpoint.pt',
    'results/checkpoints/checkpoints/3_checkpoint.pt',

    # 'results/checkpoints/checkpoints_03.09_ADAM_LCM_SYNTH_POLY2/1_checkpoint.pt',
    # 'results/checkpoints/checkpoints_03.09_ADAM_LCM_SYNTH_POLY2/2_checkpoint.pt',
    # 'results/checkpoints/checkpoints_03.09_ADAM_LCM_SYNTH_POLY2/3_checkpoint.pt'

    # 'results/checkpoints/checkpoints_04.09_ADAM_LCM_POLY2/1_checkpoint.pt',
    # 'results/checkpoints/checkpoints_04.09_ADAM_LCM_POLY2/2_checkpoint.pt',
    # 'results/checkpoints/checkpoints_04.09_ADAM_LCM_POLY2/3_checkpoint.pt',
]

plot_true = True
plot_x_eq_y = True

# plot_true = False
# plot_x_eq_y = True

mus_true = [u_true_1 @ poly_deg_2, u_true_2 @ poly_deg_2, u_true_2 @ poly_deg_2]
sigmas_true = [v_true_1 @ poly_deg_1, v_true_2 @ poly_deg_1, v_true_3 @ poly_deg_1]
# gamma_true = [z_true_1 @ poly_deg_1, z_true_2 @ poly_deg_1, z_true_3 @ poly_deg_1]
gamma_true = [z_true_1 @ poly_deg_2, z_true_2 @ poly_deg_2, z_true_3 @ poly_deg_2]

def create_plot():
    global splits, datasets

    cols_names = [f'fold {i + 1}' for i in range(len(splits))] + ['mean']

    fig, axes = plt.subplots(nrows=len(datasets), ncols=len(cols_names), figsize=(4 * len(cols_names), 3 * len(datasets)))

    for ax, col in zip(axes[0], cols_names):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], datasets):
        ax.set_ylabel(row)
    
    for col in axes:
        for ax in col:
            ax.grid()

    return fig, axes

def vis_mu():

    def get_mus(checkpoint_name):
        global poly_deg_2

        checkpoint = torch.load(checkpoint_name, map_location='cpu')
        checkpoint = checkpoint['model_state_dict']

        mus = []
        for d in range(len(datasets)):
            u = checkpoint[f'lcm.us.{d}'].numpy()
            mus.append(u @ poly_deg_2)

        return np.array(mus)

    fig, axes = create_plot()

    mus_mean = []

    for i, split in enumerate(splits):

        mus_pred = get_mus(split)

        mus_mean.append(mus_pred)
        
        for j, (ax, mu_pred) in enumerate(zip(axes[:, i], mus_pred)):
            ax.plot(ages, mu_pred[0], label='μ female')
            ax.plot(ages, mu_pred[1], label='μ male')

            if plot_true:
                mu_true = mus_true[j]
                ax.plot(ages, mu_true[0], label='μ true female')
                ax.plot(ages, mu_true[1], label='μ true male')

            if plot_x_eq_y:
                ax.plot(ages, ages, ls='--', c='black')

            ax.legend()

    mus_mean = np.array(mus_mean)
    mus_mean = np.mean(mus_mean, axis=0)
    
    for i, (mu_mean, ax) in enumerate(zip(mus_mean, axes[:, len(splits)])):
        ax.plot(ages, mu_mean[0], label='μ female')
        ax.plot(ages, mu_mean[1], label='μ male')
        
        if plot_true:
            mu_true = mus_true[i]
            ax.plot(ages, mu_true[0], label='μ true female')
            ax.plot(ages, mu_true[1], label='μ true male')

        if plot_x_eq_y:
            ax.plot(ages, ages, ls='--', c='black')

        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{plots_folder}/mu.png', dpi=150)


def vis_sigma():

    def get_sigmas(checkpoint_name):
        global poly_deg_1

        checkpoint = torch.load(checkpoint_name, map_location='cpu')
        checkpoint = checkpoint['model_state_dict']

        sigmas = []
        for d in range(len(datasets)):
            v = checkpoint[f'lcm.vs.{d}'].numpy()
            sigmas.append(v @ poly_deg_1)

        return sigmas

    fig, axes = create_plot()

    sigmas_mean = []

    for i, split in enumerate(splits):

        sigma_pred_0 = []
        sigma_pred_1 = []

        sigmas_pred = get_sigmas(split)

        sigmas_mean.append(sigmas_pred)

        for j, (ax, sigma_pred) in enumerate(zip(axes[:, i], sigmas_pred)):
            ax.plot(ages, sigma_pred[0], label='σ female')
            ax.plot(ages, sigma_pred[1], label='σ male')

            sigma_pred_0.append(sigma_pred[0])
            sigma_pred_1.append(sigma_pred[1])
            
            if plot_true:
                sigma_true = sigmas_true[j]
                ax.plot(ages, sigma_true[0], label='σ true female')
                ax.plot(ages, sigma_true[1], label='σ true male')

            ax.legend()

    sigmas_mean = np.array(sigmas_mean)
    sigmas_mean = np.mean(sigmas_mean, axis=0)
    
    for i, (sigma_mean, ax) in enumerate(zip(sigmas_mean, axes[:, len(splits)])):
        ax.plot(ages, sigma_mean[0], label='σ female')
        ax.plot(ages, sigma_mean[1], label='σ male')
        

        if plot_true:
            sigma_true = sigmas_true[i]
            ax.plot(ages, sigma_true[0], label='σ true female')
            ax.plot(ages, sigma_true[1], label='σ true male')

        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{plots_folder}/sigma.png', dpi=150)


def vis_gamma(show=False):
    global ages, poly_deg_1, poly_deg_2, splits

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def get_PgIags(checkpoint_name):

        checkpoint = torch.load(checkpoint_name, map_location='cpu')
        checkpoint = checkpoint['model_state_dict']

        PgIags = []
        
        # g_hat = np.array([-1, 1])
        for d in range(len(datasets)):
            z = checkpoint[f'lcm.zs.{d}'].numpy()
            # gamma = z @ poly_deg_1
            gamma = z @ poly_deg_2
            PgIag = sigmoid(-1 * gamma)
            
            ## Take p(ĝ=female|a,g)
            # PgIag = PgIag.reshape(2, -1)
            PgIags.append(PgIag)
        return np.array(PgIags)

    fig, axes = create_plot()

    PgIags_mean = []

    for i, split in enumerate(splits):

        PgIags = get_PgIags(split)

        PgIags_mean.append(PgIags)

        for j, (ax, PgIag) in enumerate(zip(axes[:, i], PgIags)):

            ax.plot(ages, PgIag[0], label='female')
            ax.plot(ages, PgIag[1], label='male')

            if plot_true:
                gamma = gamma_true[j]
                
                PgIag = sigmoid(-1 * gamma)
                Pg0Iag = PgIag[0]
                Pg1Iag =  PgIag[1]
                
                ax.plot(ages, Pg0Iag, label='female true')
                ax.plot(ages, Pg1Iag, label='male true')

            ax.legend()

    PgIags_mean = np.array(PgIags_mean)
    PgIags_mean = np.mean(PgIags_mean, axis=0)
    
    for i, (PgIag_mean, ax) in enumerate(zip(PgIags_mean, axes[:, len(splits)])):
        ax.plot(ages, PgIag_mean[0], label='female')
        ax.plot(ages, PgIag_mean[1], label='male')

        if plot_true:
            gamma = gamma_true[i]

            PgIag = sigmoid(-1 * gamma)

            ax.plot(ages, PgIag[0], label='true female')
            ax.plot(ages, PgIag[1], label='true male')


        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{plots_folder}/gamma.png', dpi=150)
    if show:
        plt.show()


if __name__ == "__main__":
    vis_mu()
    vis_sigma()
    vis_gamma()