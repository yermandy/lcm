from dataset.dataset import ListDataset
from torch.utils.data import DataLoader
from copy import deepcopy


class Fold():
    
    def __init__(self, net, datasets, loader_args, number, optimizer=None):

        trn_loader = DataLoader(ListDataset(datasets[0], name='trn', training=True), **loader_args)
        val_loader = DataLoader(ListDataset(datasets[1], name='val'), **loader_args)
        tst_loader = DataLoader(ListDataset(datasets[2], name='tst'), **loader_args)

        self.net = net
        self.optimizer = optimizer
        self.loaders = [trn_loader, val_loader, tst_loader]

        d = {'mae': [], 'gerr': [], 'cs5': []}
        self.results = {'trn': deepcopy(d), 'val': deepcopy(d), 'tst': deepcopy(d)}

        self.lowest_val_mae = float('inf')
        self.best_epoch = -1
        self.number = number