# MaMIClass
Classifying mammalian brains using supervised and unsupervised contrastive learning. Originally used for classifying MRI scans from the MaMI dataset (![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6376544.svg)). See [Mammalian_Brain_Classification.pdf](Mammalian_Brain_Classification.pdf) for the full report.

## Data Processing
To get access to the data, please contact me. 
To process the data using crop augmentations, run the cells under the "Data Processing" section in [mami.ipynb](mami.ipynb). More specifically, the function process_raw_images should be used. 

## Training
All training should be run through [main.py](main.py). 
For pretraining, pass the flag `--mode pretraining`, for finetuning - `--mode finetuning`. Furthermore, pass the flag `--dir DIR` to input a directory that contains the labels and processed data (default is "data"). 
The hyperparameters and all configuration for pretraining and finetuning are in [config.py](config.py). Commented out are some example configurations that have been used in the project.
For example:

    python main.py --mode finetuning

## Inference
For inference use [inference.py](inference.py). This file is responsible for outputing the losses graph, confusion matrix and latent space visualization. For example:

    python inference.py checkpoint\fine_tune_epoch_10.pt

## Trained models
Different model checkpoints are saved under [checkpoint](checkpoint).

Models trained on 5 mammalian families:
- `fine_tune_epoch_84_base.pth` - A base model trained only on the MaMI data using cross entropy loss.
- `fine_tune_epoch_83_pretrained.pth` - Pretrained on data from BHB10K, NIMH and MultiRat_rest.
- `fine_tune_epoch_77_pretrained_supcon.pth` - Taking the pretrained model and training it on the MaMI data using SupCon loss, only then finetuning it using cross entropy loss.

Model trained on all 12 available mammalian families:
- `fine_tune_epoch_199_all.pth` - Trained in the same manner as pretrained + SupCon, using batch size 32 instead of 10 on the SupCon training stage.