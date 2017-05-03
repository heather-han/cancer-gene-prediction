#!/bin/bash

# Authors:
# Heather Han (hhan16), Jiayao Wu (jwu86)
# 2 May 2017

# This script calls the main machine learning procedures and outputs the 
# accuracies of each model to stdout

# Read in file directories
data="$1"
results="$2"

# Preprocessed data files
dist_trainX=$results/distinct/train_x.txt
dist_trainY=$results/distinct/train_y.txt
dist_testX=$results/distinct/test_x.txt
dist_testY=$results/distinct/test_y.txt

conv_trainX=$results/conventional/train_x.txt
conv_trainY=$results/conventional/train_y.txt
conv_testX=$results/conventional/test_x.txt
conv_testY=$results/conventional/test_y.txt

output=$results/params

# Models for testing
distinct_trainNew=$results/params/feature_distinct/trainX.txt
distinct_testNew=$results/params/feature_distinct/testX.txt
conv_trainNew=$results/params/feature_conventional/trainX.txt
conv_testNew=$results/params/feature_conventional/testX.txt

distinct_trainNewMLP=$results/params/feature_distinct/MLPtrainX.txt
distinct_testNewMLP=$results/params/feature_distinct/MLPtestX.txt
conv_trainNewMLP=$results/params/feature_conventional/MLPtrainX.txt
conv_testNewMLP=$results/params/feature_conventional/MLPtestX.txt

distinct_trainPCA=$results/params/pca_distinct/trainX.txt
distinct_testPCA=$results/params/pca_distinct/testX.txt
conv_trainPCA=$results/params/pca_conventional/trainX.txt
conv_testPCA=$results/params/pca_conventional/testX.txt

distinct_trainMLPPCA=$results/params/pca_distinct/MLPtrainX.txt
distinct_testMLPPCA=$results/params/pca_distinct/MLPtestX.txt
conv_trainMLPPCA=$results/params/pca_conventional/MLPtrainX.txt
conv_testMLPPCA=$results/params/pca_conventional/MLPtestX.txt

# additional datasets:
addX=$results/params/feature_conventional/GPL96_570X.txt
addY=$results/testYFinal.txt

addXMLP=$results/params/feature_conventional/MLPGPL96_570X.txt

# SVM
python svmTest.py $dist_trainX $dist_testX $dist_trainY $dist_testY $conv_trainX $conv_testX $conv_trainY $conv_testY $output $distinct_trainNew $distinct_testNew $conv_trainNew $conv_testNew $distinct_trainPCA $distinct_testPCA $conv_trainPCA $conv_testPCA $addX $addY

# MLP
python mlpTest.py $dist_trainX $dist_testX $dist_trainY $dist_testY $conv_trainX $conv_testX $conv_trainY $conv_testY $output $distinct_trainNewMLP $distinct_testNewMLP $conv_trainNewMLP $conv_testNewMLP $distinct_trainMLPPCA $distinct_testMLPPCA $conv_trainMLPPCA $conv_testMLPPCA $addXMLP $addY

