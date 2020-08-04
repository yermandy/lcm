import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from resnet import resnet18, resnet50
from dataset.dataset import ListDataset, RandomDataset
from plot import plot_accuracy, plot_mae
from os import makedirs


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--workers', default=8, type=int, help='Workers number')
parser.add_argument('--cuda', default=0, type=int, help='Cuda device')
parser.add_argument('--checkpoint', default='', type=str, help='Checkpoint path')
args = parser.parse_args()

makedirs('results/models', exist_ok=True)


def encode_labels(age, gender):
    labels = []

    encoder = {}
    uid = 0
    for g in ['F', 'M']:
        for a in np.arange(1, 91, 1, dtype=np.uint8):
            encoder[(a, g)] = uid
            uid += 1

    for a, g in zip(age, gender):
        labels.append(encoder[(a, g)])

    return np.array(labels), encoder


def decode_labels(encoded_labels, decoder):
    age, gender = [], []
    for label in encoded_labels:
        a, g = decoder[label]
        age.append(a)
        gender.append(g)
    return np.array(age), np.array(gender)


def unpack_dataset(dataset_path):
    db = np.genfromtxt(dataset_path, delimiter=',', skip_header=1, dtype=str,)

    # [[1, 2, 3, 4, 5, 6, 7],  [1, 6, 2, 9, 3, 10, 8],  [7, 9, 10, 5, 8, 3, 4]]
    # [[8], [4], [6]]
    # [[9, 10], [5, 7], [2, 1]]

    ## filter dataset by age
    age = db[:, 10].astype(int)
    age_filter = np.flatnonzero((age > 0) & (age <= 90))
    db = db[age_filter]

    ## split trn, val and tst
    fold = db[:, 13].astype(int)
    get_indices = lambda x_fold: np.flatnonzero(np.isin(fold, x_fold) == True)

    trn_fold = [1, 2, 3, 4, 5, 6, 7]
    val_fold = [8]
    tst_fold = [9, 10]

    trn_idx = get_indices(trn_fold)
    val_idx = get_indices(val_fold)
    tst_idx = get_indices(tst_fold)

    paths = db[:, 0]
    boxes = db[:, [1,2,5,6]].astype(int)
    age, gender = db[:, 10].astype(int), db[:, 11]

    labels, cartesian = encode_labels(age, gender)

    datasets = []
    for idx in [trn_idx, val_idx, tst_idx]:

        datasets.append({
            'image_paths': paths[idx],
            'boxes': boxes[idx],
            'labels': labels[idx],
        })

    return datasets, cartesian


def predict(net, loader, decoder, desc=''):
    
    device = next(net.parameters()).device
    net.eval()

    total, correct, absolute_error, gerr_sum = 0, 0, 0, 0
    progress = tqdm(loader)

    age_labels = np.arange(1, 91, 1, dtype=np.float32)
    cost_matrix = np.transpose(([age_labels] * len(age_labels)))
    cost_matrix = np.abs(cost_matrix - age_labels)
    cost_matrix = torch.tensor(cost_matrix).to(device)

    gender_labels = np.array(['F', 'M'])
    zero_one_cost = np.array([0, 1], dtype=np.float32)
    zero_one_cost = torch.tensor(zero_one_cost).to(device)

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
            # absolute_error += np.abs(age_pred - age_true).sum()

            ## p(a,g|x;Θ)
            predictions = F.softmax(outputs, dim=1)

            for true_age, true_gender, pred in zip(age_true, gender_true, predictions):    

                pred = pred.reshape(2, -1)

                ## p(a|x;Θ), marginalize over gender
                PaIx = pred.sum(0)
                pred_idx = torch.argmin(PaIx @ cost_matrix)
                pred_age = age_labels[pred_idx]

                absolute_error += np.abs(pred_age - true_age)

                ## p(g|x;Θ), marginalize over age
                PgIx = pred.sum(1)
                pred_idx = torch.argmin(PgIx @ zero_one_cost)
                pred_gender = gender_labels[pred_idx]
                gerr_sum += int(true_gender != pred_gender)

            mae = absolute_error / total
            gerr = gerr_sum / total
            accuracy = correct / total * 100
            
            progress.set_description(f'{desc}')
            progress.set_postfix(accuracy=f'{accuracy:.2f}%', mae=f'{mae:.2f}', gerr=f'{gerr:.2f}')

    return accuracy, mae

def train():

    dataset_path = 'resources/agedb_28-Feb-2020_f.csv'
    datasets, encoder = unpack_dataset(dataset_path)
    decoder = {v: k for k, v in encoder.items()}

    loader_args = {
        'num_workers': args.workers,
        'batch_size': args.batch_size,
        'shuffle': True
    }


    trn_loader = DataLoader(ListDataset(datasets[0], training=True), **loader_args)
    val_loader = DataLoader(ListDataset(datasets[1]), **loader_args)
    tst_loader = DataLoader(ListDataset(datasets[2]), **loader_args)

    cudnn.benchmark = True
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # net = resnet18(num_classes=len(encoder))
    net = resnet50(num_classes=len(encoder))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            print(f'Pre-trained model: {args.checkpoint}')
            net.load_state_dict(checkpoint['model_state_dict'])

    net = net.to(device)

    epochs = 200
    lr = 0.001

    optimizer = Adam(net.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    trn_mae, val_mae, tst_mae = [], [], []

    for epoch in range(epochs):

        # acc, mae = predict(net, val_loader, decoder, 'val')
        
        net.train()
        
        mean_loss = 0
        progress = tqdm(trn_loader)
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
        
        ## calculate training accuracy and mae
        acc, mae = predict(net, trn_loader, decoder, 'trn')
        trn_mae.append(mae)

        ## calculate validation accuracy and mae
        acc, mae = predict(net, val_loader, decoder, 'val')

        val_mae.append(mae)

        ## calculate test accuracy and mae
        acc, mae = predict(net, tst_loader, decoder, 'tst')
        tst_mae.append(mae)


        plot_mae([
            [trn_mae, 'Training'], 
            [val_mae, 'Validation'],
            [tst_mae, 'Testing']
        ])
        
        checkpoint = {'model_state_dict': net.state_dict()}
        torch.save(checkpoint, f'results/models/model_{epoch + 1}.pt')

        print()

if __name__ == "__main__":
    train()