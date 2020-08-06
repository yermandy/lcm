import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from resnet import resnet18, resnet50
from dataset.dataset import ListDataset, RandomDataset, combine_datasets
from plot import plot_accuracy, plot_mae, plot_results
from os import makedirs
from copy import deepcopy
from fold import Fold
from prettytable import PrettyTable
from config import *
from encoder_decoder import encode_labels, decode_labels, create_encoder_decoder


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--workers', default=8, type=int, help='Workers number')
parser.add_argument('--cuda', default=0, type=int, help='Cuda device')
parser.add_argument('--checkpoint', default='', type=str, help='Checkpoint path')
args = parser.parse_args()

makedirs('results/checkpoints', exist_ok=True)

def create_fold_stages(db, selected_folders, encoder):
    ## filter dataset by age
    age = db[:, 10].astype(int)
    age_filter = np.flatnonzero((age >= 1) & (age <= 90))
    db = db[age_filter]

    ## split trn, val and tst
    fold = db[:, 13]
    get_indices = lambda x_fold: np.flatnonzero(np.isin(fold, x_fold) == True)

    trn_idx = get_indices(selected_folders[0])
    val_idx = get_indices(selected_folders[1])
    tst_idx = get_indices(selected_folders[2])

    paths = db[:, 0]
    boxes = db[:, [1,2,5,6]].astype(int)
    age, gender = db[:, 10].astype(int), db[:, 11]

    labels = encode_labels(age, gender, encoder)

    stages = []
    for idx in [trn_idx, val_idx, tst_idx]:

        stages.append({
            'image_paths': paths[idx],
            'boxes': boxes[idx],
            'labels': labels[idx],
        })

    return stages


def predict(net, loader, decoder, desc=''):
    
    device = next(net.parameters()).device
    net.eval()

    age_labels = np.arange(1, 91, 1, dtype=np.float32)
    cost_matrix = np.transpose(([age_labels] * len(age_labels)))
    cost_matrix = np.abs(cost_matrix - age_labels)
    cost_matrix = torch.tensor(cost_matrix).to(device)

    gender_labels = np.array(['F', 'M'])
    zero_one_cost = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32).to(device)

    total, correct, abs_err_sum, gerr_sum, cs5_sum = [0] * 5
    progress = tqdm(loader)

    with torch.no_grad():    
        for inputs, labels in progress:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            predicted = torch.max(outputs, 1)[1]

            correct += (predicted == labels).sum().item()
            total += labels.shape[0]

            labels = labels.cpu().numpy()
            age_true, gender_true = decode_labels(labels, decoder)

            ## error using cross entropy predictions
            # predicted = predicted.cpu().numpy()
            # age_pred, gender_pred = decode_labels(predicted, decoder)
            # abs_err_sum += np.abs(age_pred - age_true).sum()

            ## p(a,g|x;Θ)
            predictions = F.softmax(outputs, dim=1)

            for true_age, true_gender, pred in zip(age_true, gender_true, predictions):    

                pred = pred.reshape(2, -1)

                ## p(a|x;Θ), marginalize over gender
                PaIx = pred.sum(0)
                pred_idx = torch.argmin(PaIx @ cost_matrix)
                pred_age = age_labels[pred_idx]

                abs_err = np.abs(pred_age - true_age)
                abs_err_sum += abs_err

                cs5_sum += int(abs_err <= 5)

                ## p(g|x;Θ), marginalize over age
                PgIx = pred.sum(1)
                pred_idx = torch.argmin(PgIx @ zero_one_cost)
                pred_gender = gender_labels[pred_idx]
                gerr_sum += int(true_gender != pred_gender)

            mae = abs_err_sum / total
            gerr = gerr_sum / total
            cs5 = cs5_sum / total
            
            progress.set_description(f'{desc}')
            progress.set_postfix(gerr=f'{gerr:.2f}', cs5=f'{cs5:.2f}', mae=f'{mae:.2f}')

    return mae, gerr, cs5


def load_model(net):
    device = next(net.parameters()).device
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            print(f'Pre-trained model: {args.checkpoint}')
            net.load_state_dict(checkpoint['model_state_dict'])


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
        load_model(net)

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