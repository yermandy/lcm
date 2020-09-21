import numpy as np
from parser import args

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from model.resnet import ResNet
from tqdm import tqdm
from plot import plot_results
from os import makedirs
from fold import create_folds
from config import *
from predict import predict
from encoder_decoder import EncoderDecoder


def train():

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f'Running model {args.model} on: {device}\n')

    folds = create_folds(datasets, args, device)

    for fold in folds:
        fold.set_optimizer(Adam, lr=lr)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        epoch_results = []

        ## k-Fold Cross-Validation
        for fold in folds:
            net : ResNet = fold.net
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
                        
                        if 'lcm' in net.model_name:
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
                        'datasets': datasets,
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1, 'mae': mae, 'gerr': gerr, 'cs5': cs5,
                        'EncoderDecoder': EncoderDecoder()
                    }
                    torch.save(checkpoint, f'{results_path}/checkpoints/{fold.number}_checkpoint_best.pt')

                results[loader_name]['mae'].append(mae)
                results[loader_name]['gerr'].append(gerr)
                results[loader_name]['cs5'].append(cs5)

                fold_results.append([mae, gerr, cs5])

            plot_results(results, fold.number, results_path)

            checkpoint = {
                'model_state_dict': net.state_dict(),
                'datasets': datasets,
                'epoch': epoch + 1, 'mae': mae, 'gerr': gerr, 'cs5': cs5,
                'EncoderDecoder': EncoderDecoder()
            }

            torch.save(checkpoint, f'{results_path}/checkpoints/{fold.number}_checkpoint_last.pt')

            epoch_results.append(fold_results)
        
            print()


if __name__ == "__main__":
    
    results_path = f'results/model_{args.model}'
    
    try:
        makedirs(results_path)
    except FileExistsError:
        print(f'Folder exist: {results_path}.\n\'y\' to continue')
        input_result = input()
        if input_result != 'y':
            exit()
        print()

    makedirs(f'{results_path}/checkpoints', exist_ok=True)

    train()