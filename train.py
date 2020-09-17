import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from tqdm import tqdm
from resnet import resnet18, resnet50
from resnet_lcm import resnet50_lcm
from dataset.dataset import combine_datasets, create_fold_stages
from plot import plot_accuracy, plot_mae, plot_results
from os import makedirs
from fold import create_folds
from prettytable import PrettyTable
from config import *
from predict import predict
from encoder_decoder import EncoderDecoder


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--workers', default=8, type=int, help='Workers number')
parser.add_argument('--cuda', default=0, type=int, help='Cuda device')
parser.add_argument('--checkpoint', default='', type=str, help='Checkpoint path')
parser.add_argument('--model', default='lcm', type=str, help='Model name')
parser.add_argument('--dataset_n', default=None, type=int, help='Dataset number')
args = parser.parse_args()

makedirs('results/checkpoints', exist_ok=True)


def train():

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f'Running model on: {device}\n')

    folds = create_folds(datasets, args, device)

    for fold in folds:
        fold.set_optimizer(Adam, lr=lr)

    criterion = nn.CrossEntropyLoss()

    table = PrettyTable()
    table.field_names = ['', 'mae', 'gerr', 'cs5']

    for epoch in range(epochs):

        epoch_results = []

        ## k-Fold Cross-Validation
        for i_th_fold, fold in enumerate(folds):
            net : resnet50 = fold.net
            optimizer = fold.optimizer
            results = fold.results

            fold_results = []

            for loader in fold.loaders:
                
                loader_name = loader.dataset.name

                if loader.dataset.training:
                    net.train()
                    mean_loss = 0
                    progress = tqdm(loader, position=0)
                    for i, (inputs, labels, d) in enumerate(progress):
                        
                        optimizer.zero_grad()

                        inputs = inputs.to(device)
                        
                        if net.model_name == 'lcm':
                            outputs = net(inputs, d)
                            labels = labels.cpu().numpy()

                            ## p(â,ĝ|x,d)
                            outputs = outputs[range(len(labels)), labels]

                            ## -mean(log(p(â,ĝ|x,d)))
                            loss = outputs.clamp(1e-32, 1).log().mean().neg()
                        else:
                            outputs = net(inputs)
                            labels = labels.to(device)
                            loss = criterion(outputs, labels)

                        loss.backward()

                        optimizer.step()

                        loss = loss.item()
                        mean_loss = (i * mean_loss + loss) / (i + 1)

                        progress.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                        progress.set_postfix(loss=f'{mean_loss:.2f}')
                            
                mae, gerr, cs5 = predict(net, loader, loader_name)

                if loader_name == 'val' and mae < fold.lowest_val_mae:
                    fold.lowest_val_mae = mae
                    fold.best_epoch = epoch + 1
                    checkpoint = {
                        'model_state_dict': net.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'mae': mae, 'gerr': gerr, 'cs5': cs5,
                        'EncoderDecoder': EncoderDecoder()
                    }
                    torch.save(checkpoint, f'results/checkpoints/{fold.number}_checkpoint.pt')

                results[loader_name]['mae'].append(mae)
                results[loader_name]['gerr'].append(gerr)
                results[loader_name]['cs5'].append(cs5)

                fold_results.append([mae, gerr, cs5]) 

            plot_results(results, fold.number)

            epoch_results.append(fold_results)
        
            print()

        table.clear_rows()
        
        mean = np.mean(epoch_results, axis=0)
        std = np.std(epoch_results, axis=0)
        for stage, mu, sigma in zip(['trn', 'val', 'tst'], mean, std):
            table.add_row([stage, *[f'{m:.2f} ({s:.2f})' for m, s in zip(mu, sigma)]])

        summary = f'Epoch {epoch + 1} summary:\n{table}\n\n'
        print(summary)
        with open('results/checkpoints/summary.txt', 'a') as file:
            file.write(summary)


if __name__ == "__main__":
    train()