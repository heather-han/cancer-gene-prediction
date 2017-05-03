#!/bin/bash

# Authors:
# Heather Han (hhan16), Jiayao Wu (jwu86)
# 2 May 2017

# Bash file to run all files for the project

if [ ! -d "results" ]; then
    mkdir results
fi

data_directory='./data'
results_directory='./results'

sh process_data.sh $data_directory $results_directory
sh train_model.sh $data_directory $results_directory
sh test_model.sh $data_directory $results_directory
