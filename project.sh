#!/bin/bash

# Create text files
if [ ! -d distinct ]; then
    mkdir distinct
fi

if [ ! -d conventional ]; then
    mkdir conventional
fi

if [[ -d distinct && -d conventional ]]; then
    rm -r distinct
    rm -r conventional
    mkdir distinct
    mkdir conventional

    touch distinct/train_x.txt
    touch distinct/test_x.txt
    touch distinct/train_y.txt
    touch distinct/test_y.txt

    touch conventional/train_x.txt
    touch conventional/test_x.txt
    touch conventional/train_y.txt
    touch conventional/test_y.txt

    # Preprocess the data
    python preprocess.py GSE2034_series_matrix.txt
fi


python svm.py distinct/train_x.txt distinct/test_x.txt distinct/train_y.txt distinct/test_y.txt conventional/train_x.txt conventional/test_x.txt conventional/train_y.txt conventional/test_y.txt 

#python mlpClassifier.py distinct/train_x.txt distinct/test_x.txt distinct/train_y.txt distinct/test_y.txt conventional/train_x.txt conventional/test_x.txt conventional/train_y.txt conventional/test_y.txt 

