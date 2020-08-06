import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ListDataset(Dataset):
    def __init__(self, args, name='', training=False):

        self.image_paths = args['image_paths']
        self.boxes = args['boxes']
        self.labels = args['labels']
        self.training = training
        self.name = name

        if self.training:
            self.transform = transforms.Compose([
                transforms.Resize(140),
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
            ])


    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(f'dataset/{path}').convert('RGB')
        image = self.crop_face(image, self.boxes[index])
        return self.transform(image), self.labels[index]


    def __len__(self):
        return len(self.image_paths)


    def crop_face(self, image, box, scale=0.25):
        x1, y1, x2, y2 = box
        w = int((x2 - x1) * scale / 2)
        h = int((y2 - y1) * scale / 2)
        return image.crop((x1 - w, y1 - h, x2 + w, y2 + h))


class RandomDataset(Dataset):
    def __init__(self):
        super(RandomDataset, self).__init__()
        
        self.values = np.random.rand(10, 3, 244, 244)
        self.labels = np.random.rand(10)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


def combine_datasets(datasets_paths):
    combined_db = None
    combined_folders = []
    for dataset_id, (path, folders) in enumerate(datasets_paths):
        
        db = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=str)
        db[:, 13] = np.char.add(f'{dataset_id}_', db[:, 13])

        for folder in folders:        
            for array in folder:
                for idx, k in enumerate(array):
                    array[idx] = f'{dataset_id}_{k}'

        if combined_db is not None:
            combined_db = np.concatenate((combined_db, db))
        else:
            combined_db = db

        combined_folders.append(folders)

    combined_folders = np.array(combined_folders, dtype=object)

    combined_folds = []
    for i in range(combined_folders.shape[1]):
        fold = []
        for j in range(combined_folders.shape[2]):
            fold.append([*np.concatenate(combined_folders[:, i, j])])
        combined_folds.append(fold)

    return combined_db, combined_folds
