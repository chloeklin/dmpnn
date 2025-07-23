#!/bin/bash

# Path to dataset
chemprop train --data-path data/insulator.csv \
 --task-type regression \
 --output-dir checkpoints/insulator/ \
 --data-seed 42 \
 --pytorch-seed 42 \
 --num-replicates 1 \
 --num-folds 5 \
 --save-smiles-splits \
 --epochs 300 \
 --patience 30 \
 --loss-function mae \
 --metric mae mse r2

# Task type options: regression or classification
# Output directory will be created if it doesn't exist
# data-seed ensures reproducible splits
# num_folds specifies the number of cross-validation folds
# save-smiles-splits saves the data splits to a CSV file
# epochs is the maximum number of training epochs
# patience is for early stopping (training stops if no improvement for this many epochs)
# loss-function: mae for regression, bce for binary classification, ce for multi-class
# metrics to track: mae, mse, r2 for regression; accuracy, f1, etc. for classification