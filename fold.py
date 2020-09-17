from dataset.dataset import ListDataset
from torch.utils.data import DataLoader
from copy import deepcopy
from dataset.dataset import combine_datasets, create_fold_stages
from encoder_decoder import EncoderDecoder
from resnet import resnet50
from resnet_lcm import resnet50_lcm


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

    def set_optimizer(self, optim, **kwargs):
        self.optimizer = optim(self.net.parameters(), **kwargs)


def create_folds(datasets, args, device):
    enc_dec = EncoderDecoder()

    combined_db, combined_folds, datasets_ages = combine_datasets(datasets, args.dataset_n)

    loader_args = {
        'num_workers': args.workers,
        'batch_size': args.batch_size,
        'shuffle': True
    }

    ## each fold consists of: [[trn], [val], [tst]] dictionaries
    folds = []

    for i, selected_folders in enumerate(combined_folds):
        stages = create_fold_stages(combined_db, selected_folders)

        if args.model == 'lcm':
            net = resnet50_lcm(len(datasets), num_classes=len(enc_dec)).to(device)
        else:
            net = resnet50(num_classes=len(enc_dec)).to(device)

        folds.append(Fold(net, stages, loader_args, i + 1))

    return folds