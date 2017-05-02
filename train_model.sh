#!/bin/bash
# This script will call the main machine learning procedures and output the
# model parameters to a file

# Read in file directories
data="$1"
results="$2"
mkdir $results/params
mkdir $results/params/distinct
mkdir $results/params/conventional
mkdir $results/params/feature_distinct
mkdir $results/params/feature_conventional
mkdir $results/params/pca_distinct
mkdir $results/params/pca_conventional

dist_trainX=$results/distinct/train_x.txt
dist_trainY=$results/distinct/train_y.txt
dist_testX=$results/distinct/test_x.txt
dist_testY=$results/distinct/test_y.txt

conv_trainX=$results/conventional/train_x.txt
conv_trainY=$results/conventional/train_y.txt
conv_testX=$results/conventional/test_x.txt
conv_testY=$results/conventional/test_y.txt

genes=$results/genes.txt
gene_index=$results/params/feature_conventional/selectedFeatures.txt

addX=$results/GPL96_X.txt

# SVM

python svmTrain.py $dist_trainX $dist_testX $dist_trainY $dist_testY $conv_trainX $conv_testX $conv_trainY $conv_testY $results $addX

# MLP
python mlpTrain.py $dist_trainX $dist_testX $dist_trainY $dist_testY $conv_trainX $conv_testX $conv_trainY $conv_testY $results $addX

#findWhichFeaturesSelected
python findFeatures.py $genes $gene_index $results

