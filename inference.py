import numpy as np
import pandas as pd
import os
import joblib
from dataset import MRIDataset, CustomImageDataset
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch
from yAwareContrastiveLearning import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss, NTXenLoss, SupConLoss
from torch.nn import CrossEntropyLoss
from models.densenet import densenet121
from models.unet import UNet
import argparse
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from config import Config, PRETRAINING, FINE_TUNING, CLASSES

def get_predictions(loader, net, is_encoder=False):
    y_pred = []
    y_true = []
    for inputs, labels, paths in tqdm(loader, desc="Getting data predictions"):
        if is_encoder:
            output = net(inputs).data.cpu().numpy()
        else:
            output = torch.max(net(inputs), 1)[1].data.cpu().numpy()
        y_pred.extend(output) # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
        
    return y_pred, y_true

def get_embeddings(loader, net, unknown=False):
    embed = []
    y_true = []
            
    for inputs, labels, paths in tqdm(loader, desc="Getting data embeddings"):
        output = net(inputs, return_hidden=True).data.cpu().numpy()
        embed.extend(output) # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
        
    return embed, y_true

def plot_losses(losses):
    plt.plot(range(len(losses['train'])), losses['train'], label="train")
    plt.plot(range(len(losses['validation'])), losses['validation'], label="validation")
    plt.legend()
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.savefig('losses.png', bbox_inches = 'tight', dpi=300)
    plt.close()

def plot_confusion_matrix(loader, net):
    y_pred, y_true = get_predictions(loader, net)
    print(f'Accuracy: {accuracy_score(y_true, y_pred):.2%}')
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASSES, cmap='Blues', values_format='.2%', normalize='true')
    plt.savefig('confusion_matrix.png', bbox_inches = 'tight', dpi=300)
    plt.close()

def plot_latent_space(loader, net):
    embed, y_true = get_embeddings(loader, net)
    tsne = TSNE(n_components=2, random_state=123, n_iter=10000)
    z = tsne.fit_transform(embed) 
    scatter = plt.scatter(x=z[:,0], y=z[:,1], c=y_true)
    plt.legend(handles=scatter.legend_elements()[0], labels=CLASSES)
    plt.savefig('latent_space.png', bbox_inches = 'tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    config = Config(FINE_TUNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the checkpoint file of the model")
    parser.add_argument("--dir", type=str, default='data/',
                        help="The input directory that contains the labels and processed data. By default: data/")
    args = parser.parse_args()

    dataset_test = CustomImageDataset(config, args.dir + '/test.csv', args.dir + '/Processed/', FINE_TUNING)
    loader_test = DataLoader(dataset_test,
                              batch_size=config.batch_size,
                              pin_memory=config.pin_mem,
                              num_workers=config.num_cpu_workers
                              )

    checkpoint = torch.load(args.model_path)

    net = densenet121(mode="classifier", drop_rate=0.0, num_classes=5)
    net = torch.nn.DataParallel(net).to('cuda')
    net.load_state_dict(checkpoint['model'])

    plot_losses(checkpoint['losses'])
    plot_confusion_matrix(loader_test, net)
    plot_latent_space(loader_test, net)




