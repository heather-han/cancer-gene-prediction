#!/bin/bash
# This script will call preprocess.py on the given dataset

# Read in file directories
data="$1"
results="$2"

if [ -d $results ]; then
    rm -r $results

    # Create the relevant directories
    mkdir $results
    mkdir $results/distinct
    mkdir $results/conventional

    # Create text files
    touch $results/distinct/train_x.txt
    touch $results/distinct/test_x.txt
    touch $results/distinct/train_y.txt
    touch $results/distinct/test_y.txt

    touch $results/conventional/train_x.txt
    touch $results/conventional/test_x.txt
    touch $results/conventional/train_y.txt
    touch $results/conventional/test_y.txt

    # Preprocess the data
    python preprocess.py data/GSE2034_series_matrix.txt $results data/GSE14020-GPL96_series_matrix.txt data/GSE14020-GPL570_series_matrix.txt
    
fi