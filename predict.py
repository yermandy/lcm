import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from resnet import resnet18, resnet50
from resnet_lcm import resnet50_lcm
from dataset.dataset import combine_datasets, create_fold_stages
from plot import plot_accuracy, plot_mae, plot_results, plot_age_density, plot_age_mae
from os import makedirs
from copy import deepcopy
from fold import create_folds
from prettytable import PrettyTable
from config import *
from encoder_decoder import EncoderDecoder
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--workers', default=8, type=int, help='Workers number')
parser.add_argument('--cuda', default=0, type=int, help='Cuda device')
parser.add_argument('--checkpoints', default='', type=str, help='Path to folder with checkpoints')
parser.add_argument('--model', default='lcm', type=str, help='Model name')
parser.add_argument('--dataset_n', default=None, type=int, help='Dataset number')
args = parser.parse_args()

makedirs('results/checkpoints', exist_ok=True)



def predict(net : resnet50, loader, desc='', return_statistics=False):
    
    device = next(net.parameters()).device
    net.eval()

    age_labels = np.arange(1, 91, 1, dtype=np.float32)
    cost_matrix = np.transpose(([age_labels] * len(age_labels)))
    cost_matrix = np.abs(cost_matrix - age_labels)
    cost_matrix = torch.Tensor(cost_matrix).to(device)

    gender_labels = np.array(['F', 'M'])
    zero_one_cost = torch.Tensor([[0, 1], [1, 0]]).to(device)

    total, correct, abs_err_sum, gerr_sum, cs5_sum = [0] * 5
    progress = tqdm(loader)

    age_true_stat, age_pred_stat = [], []

    ages_distributions = []
    PagIx_stat = []

    with torch.no_grad():    
        for inputs, labels, d in progress:
            inputs = inputs.to(device)
            labels = labels.to(device)
           
            total += labels.shape[0]

            labels = labels.cpu().numpy()
            
            true_age_labels, true_gender_labels = EncoderDecoder().decode_labels(labels)

            ## error using cross entropy predictions
            # correct += (predicted == labels).sum().item()
            # predicted = torch.max(outputs, 1)[1]
            # predicted = predicted.cpu().numpy()
            # age_pred, gender_pred = decode_labels(predicted, decoder)
            # abs_err_sum += np.abs(age_pred - age_true).sum()


            if net.model_name == 'lcm':
                preds, PagIx = net(inputs, d, return_PagIx=True)

                loop = zip(true_age_labels, true_gender_labels, preds, PagIx)
            else:
                outputs = net(inputs)
                ## p(a,g|x;Θ)
                preds = F.softmax(outputs, dim=1)

                loop = zip(true_age_labels, true_gender_labels, preds)
            
            for loop_item in loop:

                if net.model_name == 'lcm':
                    true_age, true_gender, pred, PagIxi = loop_item
                else:
                    true_age, true_gender, pred = loop_item

                pred = pred.reshape(2, -1)

                ## p(â|x;Θ), marginalize over gender
                PaIx = pred.sum(0)
                pred_idx = torch.argmin(PaIx @ cost_matrix)
                pred_age = age_labels[pred_idx]

                abs_err = np.abs(pred_age - true_age)
                abs_err_sum += abs_err

                cs5_sum += int(abs_err <= 5)

                ## p(ĝ|x;Θ), marginalize over age
                PgIx = pred.sum(1)
                pred_idx = torch.argmin(PgIx @ zero_one_cost)
                pred_gender = gender_labels[pred_idx]
                gerr_sum += int(true_gender != pred_gender)

                if return_statistics:
                    age_true_stat.append(true_age)
                    age_pred_stat.append(pred_age)
                    ages_distributions.append(PaIx.cpu().numpy())

                    if net.model_name == 'lcm':
                        PagIxi = PagIxi.reshape(2, -1).sum(0)
                        PagIx_stat.append(PagIxi.cpu().numpy())

            mae = abs_err_sum / total
            gerr = gerr_sum / total
            cs5 = cs5_sum / total
            
            progress.set_description(f'{desc}')
            progress.set_postfix(gerr=f'{gerr:.2f}', cs5=f'{cs5:.2f}', mae=f'{mae:.2f}')

    if return_statistics:
        age_distribution = np.mean(ages_distributions, axis=0)
        if net.model_name == 'lcm':
            PagIx_stat = np.mean(PagIx_stat, axis=0)
            return mae, gerr, cs5, np.array(age_true_stat), np.array(age_pred_stat), age_distribution, PagIx_stat
        else:
            return mae, gerr, cs5, np.array(age_true_stat), np.array(age_pred_stat), age_distribution

    return mae, gerr, cs5


if __name__ == "__main__":

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f'Running model on: {device}\n')

    encoder_decoder = EncoderDecoder()

    folds = create_folds(datasets, args, device)

    for i, fold in enumerate(folds):
        fold.net.load_checkpoint(f'{args.checkpoints}/{i + 1}_checkpoint.pt')

    table = PrettyTable()
    table.field_names = ['', 'mae', 'gerr', 'cs5']

    prediction_summary = []
    d = {age: 0 for age in range(1, 91)}
    d = {'trn': deepcopy(d), 'val': deepcopy(d), 'tst': deepcopy(d)}
    ages_true_count = deepcopy(d)
    # ages_pred_count = deepcopy(d)
    
    d = {age: np.array([0.0, 0.0]) for age in range(1, 91)}
    d = {'trn': deepcopy(d), 'val': deepcopy(d), 'tst': deepcopy(d)}
    mae_through_ages = deepcopy(d)

    ages = np.array([0.0 for _ in range(1, 91)])
    age_distribution_dict = {'trn': deepcopy(ages), 'val': deepcopy(ages), 'tst': deepcopy(ages)}
    PagIx_dict = {'trn': deepcopy(ages), 'val': deepcopy(ages), 'tst': deepcopy(ages)}

    for fold in folds:
        net = fold.net
        # results = fold.results

        fold_results = []

        for loader in fold.loaders:
            
            loader_name = loader.dataset.name

            prediction_results = predict(net, loader, loader_name, return_statistics=True)
            
            if net.model_name == 'lcm':
                mae, gerr, cs5, ages_true_list, ages_pred_list, age_distribution, PagIx_stat = prediction_results
                PagIx_dict[loader_name] += PagIx_stat
            else:
                mae, gerr, cs5, ages_true_list, ages_pred_list, age_distribution = prediction_results

            fold_results.append([mae, gerr, cs5])
            
            age_distribution_dict[loader_name] += age_distribution

            def get_ages_count(ages_dict, ages_list):
                ages_unique, ages_count = np.unique(ages_list, return_counts=True)
                for unique, count in zip(ages_unique, ages_count):
                    ages_dict[unique] += count

            get_ages_count(ages_true_count[loader_name], ages_true_list)
        
            ## calculate mae for all ages
            maes = np.abs(ages_true_list - ages_pred_list)

            for true, mae in zip(ages_true_list, maes):
                mae_through_ages[loader_name][true] += np.array([mae, 1])

        prediction_summary.append(fold_results)
    
        print()


    for k, v in age_distribution_dict.items():
        age_distribution_dict[k] = {age: c for age, c in zip(range(1, 91), v)}

    distributions = [ages_true_count, age_distribution_dict]
    
    if net.model_name == 'lcm':
        for k, v in PagIx_dict.items():
            PagIx_dict[k] = {age: c for age, c in zip(range(1, 91), v)}
        
        distributions.append(PagIx_dict)

   ## plots

    plot_age_density(distributions, len(folds))
    plot_age_mae(mae_through_ages, len(folds))

    ## table
    table.clear_rows()
    
    mean = np.mean(prediction_summary, axis=0)
    std = np.std(prediction_summary, axis=0)
    for stage, mu, sigma in zip(['trn', 'val', 'tst'], mean, std):
        table.add_row([stage, *[f'{m:.2f} ({s:.2f})' for m, s in zip(mu, sigma)]])

    summary = f'Summary:\n{table}\n\n'
    print(summary)
    with open(f'{args.checkpoints}/prediction_summary.txt', 'a') as file:
        file.write(summary)

