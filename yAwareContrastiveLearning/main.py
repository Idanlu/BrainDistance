import numpy as np
import pandas as pd
import os
import joblib
from dataset import MRIDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch import Tensor
import torch
from yAwareContrastiveLearning import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss
from torch.nn import CrossEntropyLoss
from models.densenet import densenet121
from models.unet import UNet
import argparse
from config import Config, PRETRAINING, FINE_TUNING
from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.input_size = (1, 121, 145, 121)
        
        self.transforms = Transformer()
        self.transforms.register(Normalize(), probability=1.0)
        #self.transforms.register(Flip(), probability=0.5)
        #self.transforms.register(Blur(sigma=(0.1, 0.2)), probability=0.5)
        #self.transforms.register(Noise(sigma=(0.1, 0.2)), probability=0.5)
        self.transforms.register(Cutout(patch_size=np.ceil(np.array(self.input_size)/4)), probability=1)
        #self.transforms.register(Crop(np.ceil(0.75*np.array(self.input_size)), "random", resize=True), probability=0.5)
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image, _ = joblib.load(img_path)
        image = np.expand_dims(image, axis=0)
        np.random.seed()
        x = self.transforms(image)
        x1 = torch.nan_to_num(Tensor(x.copy()))
        #x2 = self.transforms(image)
        label = self.img_labels.iloc[idx, 1]
        #x = np.stack((x1, x2), axis=0)

        return (x1, label)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    parser.add_argument("--dir", type=str, default='.',
                        help="The input directory that contains the labels and processed data")
    args = parser.parse_args()
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING

    config = Config(mode)

    ## Fill with your target dataset
    dataset_train = CustomImageDataset(args.dir + '/train.csv', args.dir + '/Processed/')
    dataset_val = CustomImageDataset(args.dir + '/test.csv', args.dir + '/Processed/')

    loader_train = DataLoader(dataset_train,
                              batch_size=config.batch_size,
                              sampler=RandomSampler(dataset_train),
                              #collate_fn=dataset_train.collate_fn,
                              pin_memory=config.pin_mem,
                              num_workers=config.num_cpu_workers
                              )
    loader_val = DataLoader(dataset_val,
                            batch_size=config.batch_size,
                            sampler=RandomSampler(dataset_val),
                            #collate_fn=dataset_val.collate_fn,
                            pin_memory=config.pin_mem,
                            num_workers=config.num_cpu_workers
                            )
    if config.mode == PRETRAINING:
        if config.model == "DenseNet":
            net = densenet121(mode="encoder", drop_rate=0.0)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="simCLR")
        else:
            raise ValueError("Unkown model: %s"%config.model)
    else:
        if config.model == "DenseNet":
            net = densenet121(mode="classifier", drop_rate=0.0, num_classes=config.num_classes, memory_efficient=True)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="classif")
        else:
            raise ValueError("Unkown model: %s"%config.model)
    if config.mode == PRETRAINING:
        loss = GeneralizedSupervisedNTXenLoss(temperature=config.temperature,
                                              kernel='rbf',
                                              sigma=config.sigma,
                                              return_logits=True)
    elif config.mode == FINE_TUNING:
        loss = CrossEntropyLoss()

    model = yAwareCLModel(net, loss, loader_train, loader_val, config)

    if config.mode == PRETRAINING:
        model.pretraining()
    else:
        model.fine_tuning()




