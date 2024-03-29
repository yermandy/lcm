import numpy as np
import matplotlib.pyplot as plt
from os import makedirs


def _plot(args, from_epoch=1, fontsize=14, path='results/plots'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for curve, label in args['curves']:
        epochs = np.arange(from_epoch, len(curve) + from_epoch)
        ax.plot(epochs, curve, lw=2, label=label)
    
    ax.legend(fontsize=fontsize)
    ax.set_xlabel(args['x_label'], fontsize=fontsize)
    ax.set_ylabel(args['y_label'], fontsize=fontsize)
    
    path += '/plots'
    makedirs(path, exist_ok=True)
    img_path = f'{path}/{args["plot_name"]}'
    fig.savefig(img_path, dpi = 300, bbox_inches='tight')
    plt.close()


def plot_accuracy(curves: tuple, fold_number, path):
    _plot({
        'curves': curves,
        'plot_name': f'{fold_number}_accuracy.png',
        'x_label': 'Epoch',
        'y_label': 'Accuracy'
    }, path=path)


def plot_mae(curves: tuple, fold_number, path):
    _plot({
        'curves': curves,
        'plot_name': f'{fold_number}_mae.png',
        'x_label': 'Epoch',
        'y_label': 'MAE'
    }, path=path)


def plot_gerr(curves: tuple, fold_number, path):
    _plot({
        'curves': curves,
        'plot_name': f'{fold_number}_gerr.png',
        'x_label': 'Epoch',
        'y_label': 'GERR'
    }, path=path)


def plot_cs5(curves: tuple, fold_number, path):
    _plot({
        'curves': curves,
        'plot_name': f'{fold_number}_cs5.png',
        'x_label': 'Epoch',
        'y_label': 'CS5'
    }, path=path)


def plot_results(results: dict, fold_number, path):
    mae, gerr, cs5 = [], [], []
    
    stages = ('Training', 'Validation', 'Testing')
    for curve, stage in zip(results.values(), stages):
        mae.append((curve['mae'], stage))
        gerr.append((curve['gerr'], stage))
        cs5.append((curve['cs5'], stage))

    plot_mae(mae, fold_number, path)
    plot_gerr(gerr, fold_number, path)
    plot_cs5(cs5, fold_number, path)


def plot_age_density(dicts: list, folds_n, path='results/plots'):
    
    def subplot(dict, ls='-', label='', c=''):
        count = np.array(list(dict.values())) / folds_n
        count = count / count.sum()
        ax.plot(list(dict.keys()), count, ls=ls, label=label, c=c)

    fig, ax = plt.subplots(1, 1)

    # ['-', '--', ':']
    for dict, label, ls, c in zip(dicts, ['true', 'p(â)', 'p(a)'], ['-', '-', '-'], ['tab:blue', 'tab:orange', 'tab:green']):
        # subplot(dict['trn'], ls=ls, label=f'trn {label}', c='tab:blue')
        # subplot(dict['val'], ls=ls, label=f'val {label}', c='tab:orange')
        # subplot(dict['tst'], ls=ls, label=f'tst {label}', c='tab:green')
        
        subplot(dict['tst'], ls=ls, label=f'tst {label}', c=c)

    ax.legend()
    ax.set_xlabel('real age')
    ax.set_ylabel('pdf')

    path += '/plots'
    makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/age_density.png', dpi=300, bbox_inches='tight')


def plot_age_mae(dict, folds_n, path='results/plots'):

    def subplot(subdict, label):    
        
        maes_counts = np.array(list(subdict.values()))
        maes = maes_counts[:, 0] / (maes_counts[:, 1] + 1e-12)

        ages = np.array(list(subdict.keys()))

        idx = np.argsort(ages)
        ages = ages[idx]
        maes = maes[idx]

        ax.plot(ages, maes, label=label)

    fig, ax = plt.subplots(1, 1)

    subplot(dict['trn'], 'trn')
    subplot(dict['val'], 'val')
    subplot(dict['tst'], 'tst')

    ax.legend()
    ax.set_xlabel('real age')
    ax.set_ylabel('mae')
    
    path += '/plots'
    makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/age_mae.png', dpi=300, bbox_inches='tight')