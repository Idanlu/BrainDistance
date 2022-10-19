import numpy as np
import pandas as pd
import os
import joblib
from dataset import MRIDataset, CustomImageDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch
from yAwareContrastiveLearning import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss, NTXenLoss, SupConLoss
from torch.nn import CrossEntropyLoss
from models.densenet import densenet121
from models.unet import UNet
import argparse
from config import Config, PRETRAINING, FINE_TUNING

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    parser.add_argument("--dir", type=str, default='data/',
                        help="The input directory that contains the labels and processed data")
    args = parser.parse_args()
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING

    config = Config(mode)

    ## Fill with your target dataset
    dataset_train = CustomImageDataset(config, args.dir + '/train.csv', args.dir + '/Processed/', mode)
    dataset_val = CustomImageDataset(config, args.dir + '/test.csv', args.dir + '/Processed/', mode)

    loader_train = DataLoader(dataset_train,
                              batch_size=config.batch_size,
                              sampler=RandomSampler(dataset_train),
                              pin_memory=config.pin_mem,
                              num_workers=config.num_cpu_workers
                              )
    loader_val = DataLoader(dataset_val,
                            batch_size=config.batch_size,
                            sampler=RandomSampler(dataset_val),
                            pin_memory=config.pin_mem,
                            num_workers=config.num_cpu_workers
                            )
    if config.mode == PRETRAINING:
        if config.model == "DenseNet":
            net = densenet121(mode="encoder", drop_rate=0.0, memory_efficient=True)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="simCLR")
        else:
            raise ValueError("Unkown model: %s"%config.model)
    else:
        if config.model == "DenseNet":
            net = densenet121(mode="classifier", drop_rate=0.0, num_classes=config.num_classes, memory_efficient=False)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="classif")
        else:
            raise ValueError("Unkown model: %s"%config.model)
    if config.mode == PRETRAINING:
        if config.loss == 'NTXent':
            loss = NTXenLoss(temperature=config.temperature,return_logits=True)
        else:
            loss = SupConLoss(temperature=config.temperature)
    elif config.mode == FINE_TUNING:
        loss = CrossEntropyLoss()

    model = yAwareCLModel(net, loss, loader_train, loader_val, config)

    if config.mode == PRETRAINING:
        model.pretraining()
    else:
        model.fine_tuning()




