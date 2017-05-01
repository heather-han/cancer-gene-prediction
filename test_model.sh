#!/bin/bash
# This script will call the main machine learning procedures and output the
# model parameters to a file

# Read in file directories
data="$1"
results="$2"

dist_trainX=$results/distinct/train_x.txt
dist_trainY=$results/distinct/train_y.txt
dist_testX=$results/distinct/test_x.txt
dist_testY=$results/distinct/test_y.txt

conv_trainX=$results/conventional/train_x.txt
conv_trainY=$results/conventional/train_y.txt
conv_testX=$results/conventional/test_x.txt
conv_testY=$results/conventional/test_y.txt

output=$results/params

distinct_trainNew=$results/params/feature_distinct/trainX.txt
distinct_testNew=$results/params/feature_distinct/testX.txt
conv_trainNew=$results/params/feature_conventional/trainX.txt
conv_testNew=$results/params/feature_conventional/testX.txt

distinct_trainNewMLP=$results/params/feature_distinct/MLPtrainX.txt
distinct_testNewMLP=$results/params/feature_distinct/MLPtestX.txt
conv_trainNewMLP=$results/params/feature_conventional/MLPtrainX.txt
conv_testNewMLP=$results/params/feature_conventional/MLPtestX.txt


# SVM

python svmTest.py $dist_trainX $dist_testX $dist_trainY $dist_testY $conv_trainX $conv_testX $conv_trainY $conv_testY $output $distinct_trainNew $distinct_testNew $conv_trainNew $conv_testNew

# MLP
python mlpTest.py $dist_trainX $dist_testX $dist_trainY $dist_testY $conv_trainX $conv_testX $conv_trainY $conv_testY $output $distinct_trainNewMLP $distinct_testNewMLP $conv_trainNewMLP $conv_testNewMLP
# PCA
# python pca.py $results/data.txt
