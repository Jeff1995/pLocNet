# pLocNet

Protein subcellular localization prediction using graph convolutional network and protein embedding

## Directory structure

* `/preprocess` contains scripts for preprocessing data.
* `/data` contains preprocessed data and some data statistics.
* `/model` contains model definitions, scripts to run models and a performance evaluation pipeline.
* `/result` contains summary plots for model performance.

## Requirements

Scripts have been tested under the following environment:

* `python 3.6.3`
* `numpy 1.12.1`
* `scipy 1.1.0`
* `sklearn 0.19.1`
* `tensorflow 1.8.0`
* `h5py 2.7.1`
* `tqdm 4.23.4`

* `R 3.4.3`
* `rhdf5 2.22.0`
* `Biostrings 2.46.0`
* `pROC 1.12.1`
* `mccr 0.4.4`
* `dplyr 0.7.4`
* `reshape2 1.4.3`
* `ggplot2 2.2.1`
