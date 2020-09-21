import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
from encoder_decoder import EncoderDecoder
import cv2
from skimage.transform import SimilarityTransform
from copy import deepcopy

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# Allow large image files
Image.MAX_IMAGE_PIXELS = None
Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class ListDataset(Dataset):
    def __init__(self, args, name='', training=False, alignment=False, return_index=False):

        self.paths = args['paths']
        self.boxes = args['boxes']
        self.labels = args['labels']
        self.datasets = args['datasets']
        self.name = name
        self.alignment = alignment
        self.training = training
        self.return_index = return_index

        if alignment:
            self.landmarks = args['landmarks']
            if training:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
        else:
            if training:
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

        self.dest_landmarks = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ])


    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(f'dataset/{path}').convert('RGB')
        if self.alignment:
            image = self.align_face(image, self.landmarks[index])
        else:
            image = self.crop_face(image, self.boxes[index])
        if self.return_index:
            return self.transform(image), self.labels[index], self.datasets[index], index
        return self.transform(image), self.labels[index], self.datasets[index]


    def __len__(self):
        return len(self.paths)


    def crop_face(self, image, box, scale=0.25):
        x1, y1, x2, y2 = box
        w = int((x2 - x1) * scale / 2)
        h = int((y2 - y1) * scale / 2)
        return image.crop((x1 - w, y1 - h, x2 + w, y2 + h))
        

    def align_face(self, image, landmarks):
        landmarks = landmarks.reshape(5,2)
        tform = SimilarityTransform()
        tform.estimate(landmarks, self.dest_landmarks)
        M = tform.params[0:2, :]
        image = cv2.warpAffine(np.array(image), M, (112, 112))
        return Image.fromarray(image)


class RandomDataset(Dataset):
    def __init__(self):
        super(RandomDataset, self).__init__()
        
        self.values = np.random.rand(10, 3, 244, 244)
        self.labels = np.random.rand(10)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


def combine_datasets(datasets, dataset_id=None):
    combined_db = None
    combined_folders = []
    ages = []
    for id, (path, folders) in enumerate(datasets.values()):
        
        print(f'{id}: {path}')

        folders = deepcopy(folders)

        db = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=str)
        db[:, 13] = np.char.add(f'{id}_', db[:, 13])

        gender_filter = np.flatnonzero((db[:, 11] == 'F') | (db[:, 11] == "M"))
        db = db[gender_filter]

        unique_ages = np.unique(db[:, 10].astype(int))
        age_filter = np.flatnonzero((unique_ages >= 1) & (unique_ages <= 90))
        unique_ages = unique_ages[age_filter]
        ages.append(unique_ages)

        if dataset_id is not None and id != dataset_id:
            continue

        # if landmarks_path != '':
        #     landmarks = np.genfromtxt(landmarks_path, delimiter=',', dtype=int)[:, 1:]
        #     db = np.hstack((db, landmarks))

        for folder in folders:        
            for array in folder:
                for idx, k in enumerate(array):
                    array[idx] = f'{id}_{k}'

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

    return combined_db, combined_folds, ages


def create_fold_stages(db, selected_folders):
    ## filter dataset by age
    age = db[:, 10].astype(int)
    age_filter = np.flatnonzero((age >= 1) & (age <= 90))
    db = db[age_filter]

    ## split trn, val and tst
    fold = db[:, 13]
    datasets = np.array([int(dataset.split('_')[0]) for dataset in fold])
    
    get_indices = lambda x_fold: np.flatnonzero(np.isin(fold, x_fold) == True)

    trn_idx = get_indices(selected_folders[0])
    val_idx = get_indices(selected_folders[1])
    tst_idx = get_indices(selected_folders[2])

    paths = db[:, 0]
    boxes = db[:, [1,2,5,6]].astype(int)
    age, gender = db[:, 10].astype(int), db[:, 11]
    # landmarks = db[:, 14:24].astype(int)

    labels = EncoderDecoder().encode_labels(age, gender)

    stages = []
    for idx in [trn_idx, val_idx, tst_idx]:

        stages.append({
            'paths': paths[idx],
            'boxes': boxes[idx],
            'labels': labels[idx],
            # 'landmarks': landmarks[idx],
            'datasets': datasets[idx]
        })

    return stages