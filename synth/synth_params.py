import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from encoder_decoder import EncoderDecoder
from dataset.dataset import ListDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from resnet import resnet50

ages = np.arange(1, 91)

a_hat = torch.Tensor(ages)
g_hat = torch.Tensor([-1, 1])

poly_deg_1 = torch.Tensor([ages ** d for d in range(2)])
poly_deg_2 = torch.Tensor([ages ** d for d in range(3)])


def create_model(u, v, z):

    mu = (u @ poly_deg_2).view(-1, 1)
    sigma = (v @ poly_deg_1).view(-1, 1)
    gamma = (z @ poly_deg_2).view(-1, 1)

    PaIag = F.softmax((-0.5 * (mu - a_hat) ** 2) / (sigma ** 2 + 1e-8), dim=1)
    PgIag = (g_hat * gamma).sigmoid()
    PaIag_PgIag = torch.cat((PaIag * PgIag[:, 0].unsqueeze(1), PaIag * PgIag[:, 1].unsqueeze(1)), dim=1)

    return PaIag_PgIag


u1 = torch.Tensor([[0, 1.25, .007], [0, 1, .005]])
v1 = torch.Tensor([[3, .1], [.1, .1]])
g_poly = np.array([-1.1, -0.155, 0.0017])
z1 = torch.Tensor([g_poly, -g_poly])


u2 = torch.Tensor([[0, 1, .001], [0, 0.75, .001]])
v2 = torch.Tensor([[1, .25], [1, .1]])
g_poly = np.array([0, -0.155, 0.0017])
z2 = torch.Tensor([g_poly, -g_poly])


u3 = torch.Tensor([[0, 1, -.003], [0, 1, .005]])
v3 = torch.Tensor([[1, .25], [5, 0.25]])
g_poly = np.array([-1, -0.1, 0.0012])
z3 = torch.Tensor([g_poly, -g_poly])



def vis_mu(us, axes):
    
    # fig, axes = plt.subplots(1, len(us), figsize=(4 * len(us), 3 * 1))

    for u, ax in zip(us, axes):

        mu = u @ poly_deg_2

        ax.plot(ages, mu[0], label='μ female')
        ax.plot(ages, mu[1], label='μ male')
        ax.plot(ages, ages, ls='--', c='black')

        ax.legend()


def vis_sigma(vs, axes):
    
    # fig, axes = plt.subplots(1, len(vs), figsize=(4 * len(vs), 3 * 1))

    for v, ax in zip(vs, axes):

        sigma = v @ poly_deg_1

        ax.plot(ages, sigma[0], label='σ female')
        ax.plot(ages, sigma[1], label='σ male')

        ax.legend()


def vis_gamma(zs, axes):

    # fig, axes = plt.subplots(1, len(zs), figsize=(4 * len(zs), 3 * 1))

    for z, ax in zip(zs, axes):

        gamma = z @ poly_deg_2
        PgIag = (-1 * gamma).sigmoid()

        ax.plot(ages, PgIag[0], label='female')
        ax.plot(ages, PgIag[1], label='male')

        ax.legend()


def vis_params(us, vs, zs):

    fig, axes = plt.subplots(3, 3, figsize=(4 * 3, 3 * 3))

    vis_mu(us, axes[0])
    vis_sigma(vs, axes[1])
    vis_gamma(zs, axes[2])

    plt.tight_layout()
    plt.savefig('synth/params.png')


def init_loader(dataset_path):

    db = np.genfromtxt(dataset_path, delimiter=',', skip_header=1, dtype=str)

    gender_filter = np.flatnonzero((db[:, 11] == 'F') | (db[:, 11] == "M"))
    db = db[gender_filter]

    dataset_ages = db[:, 10].astype(int)
    age_filter = np.flatnonzero((dataset_ages >= 1) & (dataset_ages <= 90))
    db = db[age_filter]

    age, gender = db[:, 10].astype(int), db[:, 11]
    labels = EncoderDecoder().encode_labels(age, gender)

    paths = db[:, 0]
    boxes = db[:, [1,2,5,6]].astype(int)
    folders = db[:, 13]

    dataset_args = {
        'paths': paths,
        'boxes': boxes,
        'labels': labels,
        'datasets': folders
    }

    loader_args = {
        'num_workers': 16,
        'batch_size': 128,
    }

    list_dataset = ListDataset(dataset_args)
    loader = DataLoader(list_dataset, **loader_args)

    return loader, db


def init_model(checkpoint):
    net = resnet50(num_classes=len(EncoderDecoder()))
    net.load_checkpoint(checkpoint)
    return net


# vis_params([u1, u2, u3], [v1, v2, v3], [z1, z2, z3])

datasets = [1, 2, 3]

bias_masks = [np.arange(45), np.arange(30, 61), np.arange(44, 90)]
datasets_params = [[u1, v1, z1], [u2, v2, z2], [u3, v3, z3]]

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoint_path = 'results/checkpoints_ResNet_IMDB/1_checkpoint.pt'
model = init_model(checkpoint_path).to(device)

for dataset, bias_mask, params in zip(datasets, bias_masks, datasets_params):

    u, v, z = params
    PaIag_PgIag = create_model(u, v, z).to(device)

    ## bias
    bias = np.ones(90)
    bias[bias_mask] = 0
    bias = np.array([*bias, *bias])
    bias = torch.Tensor(bias).to(device)

    dataset_path = f'resources/imdb_synth/imdb_{dataset}.csv'
    loader, db = init_loader(dataset_path)

    progress = tqdm(loader)

    labels_180 = np.arange(180)

    synth_a = []
    synth_g = []

    for inputs, labels, d in progress:
        inputs = inputs.to(device)

        outputs = model(inputs)
        
        ## p(a,g|x;Θ)
        PagIx = F.softmax(outputs + bias, dim=1)

        for p in PagIx:

            PagIx_lcm = (PaIag_PgIag * p.unsqueeze(1)).sum(0)

            PagIx_lcm = PagIx_lcm.detach().cpu().numpy()
            
            label = np.random.choice(labels_180, p=PagIx_lcm)

            age, gender = EncoderDecoder().decode_labels([label])
            age, gender = age[0], gender[0]

            synth_a.append(age)
            synth_g.append(gender)

            
    db[:, 10] = np.array(synth_a)
    db[:, 11] = np.array(synth_g)

    np.savetxt(f'resources/imdb_synth/imdb_synth_{dataset}.csv', db, fmt='%s', delimiter=',')
    
    # softmax = torch.rand(len(ages) * 2) + bias

    # PagIx = F.softmax(softmax, dim=0)

    # PagIx_lcm = (PaIag_PgIag * PagIx.unsqueeze(1)).sum(0)

    # ymax = max(PagIx_lcm)

    # PagIx_lcm = PagIx_lcm.numpy()

    # for _ in range(100):

        # line = np.random.choice(np.arange(180), p=PagIx_lcm)
        # plt.vlines(line, ymin=0, ymax=ymax)

    # plt.plot(range(len(PagIx_lcm)), PagIx_lcm, c='red')

    # plt.show()



# PaIx_lcm = PagIx_lcm.reshape(2, -1).sum(0)
# PgIx_lcm = PagIx_lcm.reshape(2, -1).sum(1)

# '''



## bias

''' 
bias = np.ones(90)

bias[np.arange(45)] = 0

bias = np.array([*bias, *bias])

bias = bias.reshape(2, -1).sum(0)

plt.plot(range(len(bias)), bias)

# plt.show()
'''