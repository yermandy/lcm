import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from encoder_decoder import EncoderDecoder
from dataset.dataset import ListDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.resnet import resnet50, resnet18
from model.senet import se_resnet18


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


def vis_mu(us, axes):
    
    for u, ax in zip(us, axes):

        mu = u @ poly_deg_2

        ax.plot(ages, mu[0], label='μ female')
        ax.plot(ages, mu[1], label='μ male')
        ax.plot(ages, ages, ls='--', c='black')

        ax.legend()


def vis_sigma(vs, axes):
    
    for v, ax in zip(vs, axes):

        sigma = v @ poly_deg_1

        ax.plot(ages, sigma[0], label='σ female')
        ax.plot(ages, sigma[1], label='σ male')

        ax.legend()


def vis_gamma(zs, axes):

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
    net = resnet18(num_classes=len(EncoderDecoder()))
    net.load_checkpoint(checkpoint)
    return net


u1 = torch.Tensor([[0, 1.9, -0.01], [0, 0.1, 0.01]])
v1 = torch.Tensor([[1, .3], [1, 0.2]])
g_poly = np.array([-3, 0.02, 0.0001])
z1 = torch.Tensor([g_poly, -g_poly])


u2 = torch.Tensor([[1, 0.5, 0.01], [-1, 1.5, -0.01]])
v2 = torch.Tensor([[1, 0.75], [1, .1]])
g_poly = np.array([-0.7, -0.04, -0.0001])
z2 = torch.Tensor([g_poly, -g_poly])


u3 = torch.Tensor([[0, 1, 0.005], [0, 1, -0.005]])
v3 = torch.Tensor([[1, 0.01], [1, 0.1]])
g_poly = np.array([-1, -0.02, 0.0002])
z3 = torch.Tensor([g_poly, -g_poly])


vis_params([u1, u2, u3], [v1, v2, v3], [z1, z2, z3])

datasets = [1, 2, 3]

add_true = False
bias_masks = [np.arange(45), np.arange(30, 61), np.arange(44, 90)]

datasets_params = [[u1, v1, z1], [u2, v2, z2], [u3, v3, z3]]


device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = 'results/checkpoints_senet_imdb/1_checkpoint.pt'
model = init_model(checkpoint_path).to(device)

for dataset, bias_mask, params in zip(datasets, bias_masks, datasets_params):

    u, v, z = params
    PaIag_PgIag = create_model(u, v, z).to(device)

    ## bias

    if add_true:
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
        if add_true:
            PagIx = F.softmax(outputs + bias, dim=1)
        else:
            PagIx = F.softmax(outputs, dim=1)

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

    np.savetxt(f'resources/imdb_synth/imdb_synth_{dataset}_senet.csv', db, fmt='%s', delimiter=',')