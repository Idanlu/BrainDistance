from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip
import os
import joblib
from config import PRETRAINING
from torch import Tensor

class MRIDataset(Dataset):

    def __init__(self, config, training=False, validation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert training != validation

        self.transforms = Transformer()
        self.config = config
        self.transforms.register(Normalize(), probability=1.0)

        if config.tf == "all_tf":
            self.transforms.register(Flip(), probability=0.5)
            self.transforms.register(Blur(sigma=(0.1, 1)), probability=0.5)
            self.transforms.register(Noise(sigma=(0.1, 1)), probability=0.5)
            self.transforms.register(Cutout(patch_size=np.ceil(np.array(config.input_size)/4)), probability=0.5)
            self.transforms.register(Crop(np.ceil(0.75*np.array(config.input_size)), "random", resize=True),
                                     probability=0.5)

        elif config.tf == "cutout":
            self.transforms.register(Cutout(patch_size=np.ceil(np.array(config.input_size)/4)), probability=1)

        elif config.tf == "crop":
            self.transforms.register(Crop(np.ceil(0.75*np.array(config.input_size)), "random", resize=True),
                                     probability=1)

        if training:
            self.data = np.load(config.data_train)
            self.labels = pd.read_csv(config.label_train)
        elif validation:
            self.data = np.load(config.data_val)
            self.labels = pd.read_csv(config.label_val)

        assert self.data.shape[1:] == tuple(config.input_size), "3D images must have shape {}".\
            format(config.input_size)

    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)

        return (list_x, list_y)

    def __getitem__(self, idx):

        # For a single input x, samples (t, t') ~ T to generate (t(x), t'(x))
        np.random.seed()
        x1 = self.transforms(self.data[idx])
        x2 = self.transforms(self.data[idx])
        labels = self.labels[self.config.label_name].values[idx]
        x = np.stack((x1, x2), axis=0)

        return (x, labels)

    def __len__(self):
        return len(self.data)

class CustomImageDataset(Dataset):
    def __init__(self, config, annotations_file, img_dir, mode):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.mode = mode

        self.transforms = Transformer()
        self.transforms.register(Normalize(), probability=1.0)
        #self.transforms.register(Flip(), probability=0.5)
        #self.transforms.register(Blur(sigma=(0.1, 0.2)), probability=0.5)
        #self.transforms.register(Noise(sigma=(0.1, 0.2)), probability=0.5)
        self.transforms.register(Cutout(patch_size=np.ceil(np.array(config.input_size)/2)), probability=1)
        #self.transforms.register(Crop(np.ceil(0.75*np.array(self.input_size)), "random", resize=True), probability=0.5)
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image, _ = joblib.load(img_path)
        image = np.expand_dims(image, axis=0)
        np.random.seed()
        if self.mode == PRETRAINING:
            x1 = self.transforms(image)
            x1 = torch.nan_to_num(Tensor(x1.copy()))
            x2 = self.transforms(image)
            x2 = torch.nan_to_num(Tensor(x2.copy()))
            x = np.stack((x1, x2), axis=0)
        else:
            #x = self.transforms(image)
            x = image
            x = torch.nan_to_num(Tensor(x.copy()))
        label = self.img_labels.iloc[idx, 2] # take the family index
        

        return (x, label, self.img_labels.iloc[idx, 0])