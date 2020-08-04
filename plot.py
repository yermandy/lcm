import numpy as np
import matplotlib.pyplot as plt
from os import makedirs


def _plot(args, from_epoch=1, fontsize = 14):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for curve, label in args['curves']:
        epochs = np.arange(from_epoch, len(curve) + from_epoch)
        ax.plot(epochs, curve, lw=2, label=label)
    
    ax.legend(fontsize=fontsize)
    ax.set_xlabel(args['x_label'], fontsize=fontsize)
    ax.set_ylabel(args['y_label'], fontsize=fontsize)
    
    path = 'results/plots'
    makedirs(path, exist_ok=True)
    img_path = f'{path}/{args["plot_name"]}'
    fig.savefig(img_path, dpi = 300, bbox_inches='tight')
    plt.close()


def plot_accuracy(curves: tuple, fold_number):
    _plot({
        'curves': curves,
        'plot_name': f'{fold_number}accuracy.png',
        'x_label': 'Epoch',
        'y_label': 'Accuracy'
    })


def plot_mae(curves: tuple, fold_number):
    _plot({
        'curves': curves,
        'plot_name': f'{fold_number}_mae.png',
        'x_label': 'Epoch',
        'y_label': 'MAE'
    })


def plot_gerr(curves: tuple, fold_number):
    _plot({
        'curves': curves,
        'plot_name': f'{fold_number}_gerr.png',
        'x_label': 'Epoch',
        'y_label': 'GERR'
    })


def plot_cs5(curves: tuple, fold_number):
    _plot({
        'curves': curves,
        'plot_name': f'{fold_number}_cs5.png',
        'x_label': 'Epoch',
        'y_label': 'CS5'
    })


def plot_results(results: dict, fold_number):
    mae, gerr, cs5 = [], [], []
    
    stages = ('Training', 'Validation', 'Testing')
    for curve, stage in zip(results.values(), stages):
        mae.append((curve['mae'], stage))
        gerr.append((curve['gerr'], stage))
        cs5.append((curve['cs5'], stage))

    plot_mae(mae, fold_number)
    plot_gerr(gerr, fold_number)
    plot_cs5(cs5, fold_number)