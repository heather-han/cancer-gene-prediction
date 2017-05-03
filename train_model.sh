#!/bin/bash

# Authors:
# Heather Han (hhan16), Jiayao Wu (jwu86)
# 2 May 2017

# This script will call the main machine learning procedures and output the
# model parameters to a file

# Read in file directories
data="$1"
results="$2"

# Create the necessary directories
mkdir $results/params
mkdir $results/params/distinct
mkdir $results/params/conventional
mkdir $results/params/feature_distinct
mkdir $results/params/feature_conventional
mkdir $results/params/pca_distinct
mkdir $results/params/pca_conventional

# Get the locations to store the models
dist_trainX=$results/distinct/train_x.txt
dist_trainY=$results/distinct/train_y.txt
dist_testX=$results/distinct/test_x.txt
dist_testY=$results/distinct/test_y.txt

conv_trainX=$results/conventional/train_x.txt
conv_trainY=$results/conventional/train_y.txt
conv_testX=$results/conventional/test_x.txt
conv_testY=$results/conventional/test_y.txt

# Array of the identity of the gene at each position
genes=$results/genes.txt
gene_index=$results/params/feature_conventional/selectedFeatures.txt

# Additional test file
addX=$results/testXFinal.txt

# SVM
python svmTrain.py $dist_trainX $dist_testX $dist_trainY $dist_testY $conv_trainX $conv_testX $conv_trainY $conv_testY $results $addX

# MLP
python mlpTrain.py $dist_trainX $dist_testX $dist_trainY $dist_testY $conv_trainX $conv_testX $conv_trainY $conv_testY $results $addX

# Find which features are significant
python findFeatures.py $genes $gene_index $results

