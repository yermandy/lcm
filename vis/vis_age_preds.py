import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from resnet import resnet50
from resnet_lcm import resnet50_lcm
from encoder_decoder import encode_labels, decode_labels, create_encoder_decoder
from dataset.dataset import ListDataset
from torch.utils.data import DataLoader
from PIL import Image


parser = argparse.ArgumentParser(description='Visualization')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--workers', default=8, type=int, help='Workers number')
parser.add_argument('--cuda', default=0, type=int, help='Cuda device')
parser.add_argument('--checkpoint', default='', type=str, help='Path to folder with checkpoints')
parser.add_argument('--model', default='lcm', type=str, help='Model name')
parser.add_argument('--dataset_id', default=0, type=int, help='Dataset number')
args = parser.parse_args()



dataset_path = 'resources/inet.csv'
lcm_checkpoint = 'results/checkpoints/checkpoints/1_checkpoint.pt'
resnet50_checkpoint = 'results/checkpoints/checkpoints_06.08_ADAM/1_checkpoint.pt'

encoder, decoder = create_encoder_decoder()


def init_loader(dataset_path, folders=['9','10']):
    dataset = np.genfromtxt(dataset_path, delimiter=',', dtype=str)

    mask = np.full(len(dataset), 0, dtype=bool)
    for folder in folders:
        mask = np.bitwise_or(dataset[:, 13] == folder, mask)
    
    dataset = dataset[mask]

    age = dataset[:, 10].astype(int)
    age_filter = np.flatnonzero((age >= 1) & (age <= 90))
    dataset = dataset[age_filter]

    gender_filter = np.flatnonzero((dataset[:, 11] == 'F') | (dataset[:, 11] == "M"))
    dataset = dataset[gender_filter]

    paths = dataset[:, 0]
    boxes = dataset[:, [1,2,5,6]].astype(int)
    age = dataset[:, 10].astype(int)
    gender = dataset[:, 11]

    labels = encode_labels(age, gender, encoder)

    datasets = np.repeat(args.dataset_id, len(labels))

    list_dataset_args = {
        'paths': paths,
        'boxes': boxes,
        'labels': labels,
        'datasets': datasets
    }

    loader_args = {
        'num_workers': args.workers,
        'batch_size': args.batch_size,
        'shuffle': True
    }

    list_dataset = ListDataset(list_dataset_args, name='tst', return_index=True)
    loader = DataLoader(list_dataset, **loader_args)

    return loader


def init_model(checkpoint, model):
    if model == 'lcm':
        net = resnet50_lcm(12, num_classes=len(encoder))
        net.load_checkpoint(checkpoint)
    elif model == 'resnet50':
        net = resnet50(num_classes=len(encoder))
        net.load_checkpoint(checkpoint)
    return net


if __name__ == "__main__":
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    loader = init_loader(dataset_path)
    paths = loader.dataset.paths
    
    model = init_model(resnet50_checkpoint, 'resnet50').to(device)
    model_lcm = init_model(lcm_checkpoint, 'lcm').to(device)

    images = 50
    fig, axes = plt.subplots(nrows=images, ncols=2, figsize=(4 * 2, 3 * images))

    ages = np.arange(1, 91, dtype=np.float32)
    #! Database ages
    database_ages = np.arange(1, 91, dtype=np.float32)

    cost_matrix = np.transpose(([ages] * len(ages)))
    cost_matrix = np.abs(cost_matrix - ages)
    cost_matrix = torch.Tensor(cost_matrix).to(device)

    for i, (inputs, labels, d, index) in enumerate(loader):
        print(paths[index])

        if i >= images:
            break

        inputs = inputs.to(device)

        ## LCM
        lcm_preds, PagIx = model_lcm(inputs, d, return_PagIx=True)

        lcm_preds = lcm_preds[0]
        lcm_preds = lcm_preds.reshape(2, -1).sum(0)
        lcm_idx = torch.argmin(lcm_preds @ cost_matrix)
        lcm_age = ages[lcm_idx]
        lcm_preds = lcm_preds.detach().cpu().numpy()
        
        PagIx = PagIx[0]
        PaIx = PagIx.reshape(2, -1).sum(0)
        PaIx_idx = torch.argmin(PaIx @ cost_matrix)
        PaIx_age = ages[PaIx_idx]
        PaIx = PaIx.detach().cpu().numpy()

        ## ResNet50
        resnet50_preds = model(inputs)
        ## p(a,g|x;Θ)
        resnet50_preds = F.softmax(resnet50_preds, dim=1)
        resnet50_preds = resnet50_preds[0]
        resnet50_preds = resnet50_preds.reshape(2, -1).sum(0)

        resnet50_idx = torch.argmin(resnet50_preds @ cost_matrix)
        resnet50_age = ages[resnet50_idx]

        resnet50_preds = resnet50_preds.detach().cpu().numpy()


        image = Image.open(f'dataset/{paths[index]}').convert('RGB')

        box = loader.dataset.boxes[index]
        image = loader.dataset.crop_face(image, box)

        axes[i, 0].imshow(image)

        axes[i, 1].plot(database_ages, PaIx, label='LCM: p(a)')
        axes[i, 1].plot(ages, lcm_preds, label='LCM: p(â)')
        axes[i, 1].plot(ages, resnet50_preds, label='ResNet: p(a)')

        ## true age
        true_age, true_gender = decode_labels(labels.detach().cpu().numpy(), decoder)
        true_age = true_age[0]

        axes[i, 1].axvline(PaIx_age, label=f'p(a|x): {int(PaIx_age)} ({int(abs(PaIx_age - true_age))})', c='red', lw=1)
        axes[i, 1].axvline(lcm_age, label=f'p(â|x): {int(lcm_age)} ({int(abs(lcm_age - true_age))})', c='magenta', lw=1)
        axes[i, 1].axvline(true_age, label=f'true age: {int(true_age)}', c='purple', lw=1)
        axes[i, 1].axvline(resnet50_age, label=f'ResNet age: {resnet50_age} ({int(abs(resnet50_age - true_age))})', c='brown', lw=1)



        axes[i, 1].legend()


    plt.tight_layout()
    plt.savefig('vis.png', dpi=150)


    # print(prediction_results)

