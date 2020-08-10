import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
from encoder_decoder import encode_labels
import cv2
from skimage.transform import SimilarityTransform

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ListDataset(Dataset):
    def __init__(self, args, name='', training=False, alignment=True):

        self.paths = args['paths']
        self.boxes = args['boxes']
        self.labels = args['labels']
        self.name = name
        self.alignment = alignment
        self.training = training

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
        return self.transform(image), self.labels[index]


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


def combine_datasets(datasets):
    combined_db = None
    combined_folders = []
    for dataset_id, (path, landmarks_path, folders) in enumerate(datasets):
        
        db = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=str)
        landmarks = np.genfromtxt(landmarks_path, delimiter=',', dtype=int)[:, 1:]
        db[:, 13] = np.char.add(f'{dataset_id}_', db[:, 13])

        db = np.hstack((db, landmarks))

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


def create_fold_stages(db, selected_folders, encoder):
    ## filter dataset by age
    age = db[:, 10].astype(int)
    age_filter = np.flatnonzero((age >= 1) & (age <= 90))
    db = db[age_filter]

    ## split trn, val and tst
    fold = db[:, 13]
    get_indices = lambda x_fold: np.flatnonzero(np.isin(fold, x_fold) == True)

    trn_idx = get_indices(selected_folders[0])
    val_idx = get_indices(selected_folders[1])
    tst_idx = get_indices(selected_folders[2])

    paths = db[:, 0]
    boxes = db[:, [1,2,5,6]].astype(int)
    age, gender = db[:, 10].astype(int), db[:, 11]
    landmarks = db[:, 14:].astype(int)

    labels = encode_labels(age, gender, encoder)

    stages = []
    for idx in [trn_idx, val_idx, tst_idx]:

        stages.append({
            'paths': paths[idx],
            'boxes': boxes[idx],
            'labels': labels[idx],
            'landmarks': landmarks[idx]
        })

    return stages