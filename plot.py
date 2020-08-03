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

def plot_accuracy(accuracies):

    args = {
        'curves': accuracies,
        'plot_name': 'accuracy.png',
        'x_label': 'Epoch',
        'y_label': 'Accuracy'
    }

    _plot(args)


def plot_mae(maes, from_epoch=1):

    args = {
        'curves': maes,
        'plot_name': 'mae.png',
        'x_label': 'Epoch',
        'y_label': 'MAE'
    }

    _plot(args)