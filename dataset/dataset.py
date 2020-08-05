import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from warnings import filterwarnings


ImageFile.LOAD_TRUNCATED_IMAGES = True
filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


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
        
        import numpy as np
        self.values = np.random.rand(10, 3, 244, 244)
        self.labels = np.random.rand(10)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]

