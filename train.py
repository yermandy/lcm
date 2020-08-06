import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from resnet import resnet18, resnet50
from dataset.dataset import ListDataset, RandomDataset, combine_datasets, create_fold_stages
from plot import plot_accuracy, plot_mae, plot_results
from os import makedirs
from copy import deepcopy
from fold import Fold
from prettytable import PrettyTable
from config import *
from encoder_decoder import encode_labels, decode_labels, create_encoder_decoder
from predict import predict


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--workers', default=8, type=int, help='Workers number')
parser.add_argument('--cuda', default=0, type=int, help='Cuda device')
parser.add_argument('--checkpoint', default='', type=str, help='Checkpoint path')
args = parser.parse_args()

makedirs('results/checkpoints', exist_ok=True)


def train():

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f'Running model on: {device}\n')

    loader_args = {
        'num_workers': args.workers,
        'batch_size': args.batch_size,
        'shuffle': True
    }

    encoder, decoder = create_encoder_decoder()

    ## concatenate datasets
    combined_db, combined_folds = combine_datasets(datasets)

    folds = []
    for number, selected_folders in enumerate(combined_folds):
        ## each fold consists of: [trn, val, tst] dictionaries
        stages = create_fold_stages(combined_db, selected_folders, encoder)
        
        net = resnet50(num_classes=len(encoder)).to(device)
        if args.checkpoint:
            net.load_checkpoint(args.checkpoint)

        optimizer = Adam(net.parameters(), lr=lr)

        folds.append(Fold(net, optimizer, stages, loader_args, number + 1))

    criterion = nn.CrossEntropyLoss()

    table = PrettyTable()
    table.field_names = ['', 'mae', 'gerr', 'cs5']

    for epoch in range(epochs):

        epoch_results = []

        ## k-Fold Cross-Validation
        for i_th_fold, fold in enumerate(folds):
            fold : Fold = fold

            net = fold.net
            optimizer = fold.optimizer
            results = fold.results

            fold_results = []

            for loader in fold.loaders:
                
                loader_name = loader.dataset.name

                if loader.dataset.training:
                    net.train()
                    mean_loss = 0
                    progress = tqdm(loader, position=0)
                    for i, (inputs, labels) in enumerate(progress):
                        
                        optimizer.zero_grad()

                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = net(inputs)    

                        loss = criterion(outputs, labels)
                        loss.backward()

                        optimizer.step()

                        loss = loss.item()
                        mean_loss = (i * mean_loss + loss) / (i + 1)

                        progress.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                        progress.set_postfix(loss=f'{mean_loss:.2f}')
                            
                mae, gerr, cs5 = predict(net, loader, decoder, loader_name)

                if loader_name == 'val' and mae < fold.lowest_val_mae:
                    fold.lowest_val_mae = mae
                    fold.best_epoch = epoch + 1
                    checkpoint = {
                        'model_state_dict': net.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'mae': mae, 'gerr': gerr, 'cs5': cs5,
                        'encoder': encoder,
                        'decoder': decoder,
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