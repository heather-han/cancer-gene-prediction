# Heather Han (hhan16)
# Jiayao Wu (jwu86)
# 29 April 2017

# python version: 2.7
import sys
import numpy as np
''' Script used to find which features are selected '''

#a list of genes
genes = np.genfromtxt(sys.argv[1],delimiter=' ', dtype=str)

#a list of important genes' indexes
gene_index = np.genfromtxt(sys.argv[2],delimiter=' ')
results = sys.argv[3]

selectedGenes = []
for i in range(0, len(genes)):
	if i in gene_index:
		selectedGenes.append(genes[i])

np.savetxt(results+'/selectedGenes.txt', selectedGenes, fmt='%s')
