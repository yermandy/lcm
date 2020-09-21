import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt
from model.resnet_lcm import resnet50lcm
from vis_age_preds import init_loader
from config import folders_10, folders_3, folders_8
from encoder_decoder import EncoderDecoder
from fold import create_folds


parser = argparse.ArgumentParser(description='Bias')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--workers', default=8, type=int, help='Workers number')
parser.add_argument('--cuda', default=0, type=int, help='Cuda device')
parser.add_argument('--checkpoint', default='', type=str, help='Checkpoint path')
parser.add_argument('--model', default='lcm', type=str, help='Model name')
parser.add_argument('--dataset_id', default=None, type=int, help='Dataset number')
args = parser.parse_args()


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(4 * 3, 3))

ages = np.arange(1, 91)

databases = []

datasets = [
    # ('resources/agedb.csv', '', folders_10),
    # ('resources/morph.csv', '', folders_10),
    # ('resources/appa_real.csv', '', folders_3)

    ('resources/imdb_synth/imdb_synth_1.csv', '', folders_8),
    ('resources/imdb_synth/imdb_synth_2.csv', '', folders_8),
    ('resources/imdb_synth/imdb_synth_3.csv', '', folders_8),
]

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

folds = create_folds(datasets, args, device)

loaders = [fold.loaders[2] for fold in folds]


true_mean = []

# for checkpoint_id, fold in zip(range(1, 4), folds):

for ax, dataset in zip(axes, datasets):
    path = dataset[0]
    db = np.genfromtxt(path, delimiter=',')

    folders = dataset[2]
    split = 0
    folders = folders[split][0]
    
    mask = np.full(len(db), 0, dtype=bool)
    for folder in folders:
        mask = np.bitwise_or(db[:, 13] == folder, mask)
    db = db[mask]
    
    db_ages = db[:, 10]
    age_filter = np.flatnonzero((db_ages >= 1) & (db_ages <= 90))
    db_ages = db_ages[age_filter]
    unique_ages, counts = np.unique(db_ages, return_counts=True)
    
    age_distribution = {age: 0 for age in range(1, 91)}

    for u, c in zip(unique_ages, counts):
        age_distribution[u] = c

    counts = np.array(list(age_distribution.values()))
    counts = counts / counts.sum()
    
    ax.plot(ages, counts, label='true distribution')
    ax.legend()


bias_mean = []

for checkpoint_id, fold in zip(range(1, 2), folds):

    # bias_mean = []
    net = fold.net

    checkpoint_name = f'results/checkpoints/checkpoints/{checkpoint_id}_checkpoint.pt'
    net.load_checkpoint(checkpoint_name)

    model_state_dict = net.state_dict()

    biases = []

    for i, dataset in enumerate(datasets):
        bias = model_state_dict[f'mbl.biases.{i}'].detach().numpy()
        bias = bias.reshape(2, -1)
        bias = bias.sum(0)
        bias = np.exp(bias)
        bias /= bias.sum()
        biases.append(bias)
        
    bias_mean.append(biases)

bias_means = np.mean(bias_mean, axis=0)
# print(bias_mean[0])
# axes[0].plot(ages, bias_mean[0])
for ax, bias_mean in zip(axes, bias_means):
    ax.plot(ages, bias_mean, label='dataset bias')
    ax.legend()
        

# for i, db in enumerate(databases):

#     bias_mean = []
#     for checkpoint_id in range(1, 4):
        
#         checkpoint_name = f'results/checkpoints/checkpoints/{checkpoint_id}_checkpoint.pt'
        
#         net = resnet50lcm(len(databases), num_classes=len(encoder))
#         net.load_checkpoint(checkpoint_name)

#         # init_loader(db, ['9, 10'])
#         # for 

        
#         bias = np.exp(bias)
#         bias = bias.sum(0)

#         bias /= bias.sum()
        
#         bias_mean.append(bias)

#     bias_mean = np.mean(bias_mean, axis=0)

#     axes[i].plot(ages, bias_mean)


plt.show()