# Pheno-Deep Counter

PhenoDC is a multi-modal deep neural network architecture for leaf counting. This architecture is able to count leaves from top-view rosette-shaped plants, using multi-modal data. 
In our experiments, we used up to three modalities (RGB, near-infrared, and fluorescence).




## Installation

1. The package should work on Python 2.7.x. We advise to install a [miniconda](https://conda.io/miniconda.html) enviroment
2. Install the following dependencies, following the directions provided according to your platform and requirements:
    - numpy
    - [Tensorflow](https://www.tensorflow.org/) (1.3 or later)
    - [Keras](https://keras.io/) (==2.1.6)
    - PILLOW
    - XlsxWriter
    - xlrd
    - scipy
    - h5py 
3. `git clone https://tuttoweb@bitbucket.org/tuttoweb/pheno-deep-counter.git` 

## Downloading Pre-trained Networks

We release the pre-trained models on the following datasets:
    - CVPPP
    - Multi-modal imagery

They can be dowloaded at: http://www.valeriogiuffrida.academy/sites/default/files/phenodc_trained_models.tar.gz

## Quick start

You can execute the file ./train_mm.sh (only on Mac/Linux).

## Under the hood

### main.py

Here you can find an example how to get and train a multi-modal deep neural network with three modalities.

## main_cvppp.py

In this file, we show how to train the network with only one modality (RGB) on the CVPPP 2017 dataset (we do not redistribuite the dataset here!)

## How to use the models

Each file accepts parameters. For a complete list of parameters, use `--help`.

In order to fine-tune with a specific dataset, you can use `--model` parameter followed by the path of the pre-trained weights

