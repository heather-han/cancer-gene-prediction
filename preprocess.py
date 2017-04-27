# Heather Han (hhan16)
# Jiayao Wu (jwu86)
# 26 April 2017

# usage: python preprocess.py GSE2034_series_matrix.txt
# python version: 2.7

import sys
import numpy as np
from sklearn import preprocessing

# Data samples in the file:
# 180 samples without metastasis to distant locations
# 106 samples with metastasis to distanct locations
# of the distant locations, the location we're tracking is bone

''' Read relevant data from file '''

# read information about the metastasis status
metastasis = np.genfromtxt(sys.argv[1],dtype=str,delimiter='\t',skip_header=35,max_rows=1)[1:]

# extract int values from metastasis
met_status = [int(i[-2:-1]) for i in metastasis]

# print metastasis information
np.savetxt('metastasis.txt', met_status)

# read sample and feature data
data = np.genfromtxt(sys.argv[1],delimiter='\t',skip_header=55,skip_footer=1)[:,1:]


''' Normalize the data '''
data_scaled = preprocessing.scale(data)

# print normalized data
# print data_scaled
np.savetxt('preprocessed_data.txt', data_scaled)