# Heather Han (hhan16)
# Jiayao Wu (jwu86)
# 26 April 2017

# usage: python preprocess.py GSE2034_series_matrix.txt
# python version: 2.7

import sys
import numpy as np
from sklearn import preprocessing


def main():
	# Data samples in the file:
	# 180 samples without metastasis to distant locations
	# 106 samples with metastasis to distanct locations
	# of the distant locations, the location we're tracking is bone

	''' Read relevant data from file '''

	# read information about the metastasis status from GSE2034_series_matrix
	metastasis = np.genfromtxt(sys.argv[1],dtype=str,delimiter='\t',skip_header=35,max_rows=1)[1:]

	# extract int values from metastasis
	met_status = [int(i[-2:-1]) for i in metastasis]

	# read sample and feature data
	data = np.genfromtxt(sys.argv[1],delimiter='\t',skip_header=55,skip_footer=1)[:,1:]

	genes = np.genfromtxt(sys.argv[1],dtype=str,delimiter='\t',skip_header=55,skip_footer=1)[:,0]
		
	''' remove 5 genes from data and genes in order to be consistent with
		the additional datasets '''

	genes = np.delete(genes, [22269, 22270, 22271, 22272, 22273, 22274], 0)
	data = np.delete(data, [22269, 22270, 22271, 22272, 22273, 22274], 0)
	
	# read results directory path
	results = sys.argv[2]

	'''save the genes'''
	np.savetxt(results+'/genes.txt', genes, fmt='%s')

	''' Normalize the data '''
	plus_one = [i+1 for i in data]
	data_log = [np.log2(i) for i in plus_one]
	data_scaled = preprocessing.scale(data_log, axis=1)
	np.savetxt(results+'/data.txt', data_scaled)

	''' perform distinct partition '''
	train_x, test_x, train_y, test_y = distinctPartion(data_scaled, met_status)
	np.savetxt(results+'/distinct/train_x.txt', train_x)
	np.savetxt(results + '/distinct/test_x.txt', test_x)
	np.savetxt(results + '/distinct/train_y.txt', train_y)
	np.savetxt(results + '/distinct/test_y.txt', test_y)

	''' perform conventional partition '''
	train_x, test_x, train_y, test_y = conventionalPartition(data_scaled, met_status)
	np.savetxt(results + '/conventional/train_x.txt', train_x)
	np.savetxt(results + '/conventional/test_x.txt', test_x)
	np.savetxt(results + '/conventional/train_y.txt', train_y)
	np.savetxt(results + '/conventional/test_y.txt', test_y)

	''' process additional datasets '''
	''' process GSE14020-GPL96_series_matrix'''
	data2 = np.genfromtxt(sys.argv[3],delimiter='\t',skip_header=70,skip_footer=1)[:,1:]
	
	''' remove 5 genes from data2 in order to be consistent with
		the additional datasets '''
	data2 = np.delete(data2, [22269, 22270, 22271, 22272, 22273, 22274], 0)

	''' normalize the data2 ''' 
	plus_one2 = [i+1 for i in data2]
	data_log2 = [np.log2(i) for i in plus_one2]
	data_scaled2 = preprocessing.scale(data_log2, axis=1)
	# np.savetxt(results+'/GPL96_X.txt', data_scaled2.T) not used


	metastasis2 = np.genfromtxt(sys.argv[3],dtype=str,delimiter='\t',skip_header=36,max_rows=1)[1:]
	met_status2 = []
	for i in metastasis2:
		#only select bone metastasis
		if 'Bone' in i:
			met_status2.append(1)
		else:
			met_status2.append(0)
	# np.savetxt(results+'/GPL96_Y.txt', met_status2) not used

	''' process GSE14020-GPL570_series_matrix '''
	data3 = np.genfromtxt(sys.argv[4],delimiter='\t',skip_header=70,skip_footer=1)[:,1:]
	(_, samples3) = data3.shape
	data3Genes = np.genfromtxt(sys.argv[4],dtype=str,delimiter='\t',skip_header=70,skip_footer=1)[:,0]
	(row_, col_) = data_scaled2.shape
	data3New = np.zeros((row_, samples3))
	
	''' since GSE14020-GPL570 has more features than needed, need to select
	 	the ones that we need '''
	geneList = genes.tolist()
	j = 0
	for i in data3Genes:
		if i in geneList:
			idx = geneList.index(i)
			data3New[j,:] = data3[idx,:]
			j += 1

	''' normalize data3 '''
	plus_one3 = [i+1 for i in data3New]
	data_log3 = [np.log2(i) for i in plus_one3]
	data_scaled3 = preprocessing.scale(data_log3, axis=1)
	# np.savetxt(results+'/GPL570_X.txt', data_scaled3.T) not used

	metastasis3 = np.genfromtxt(sys.argv[4],dtype=str,delimiter='\t',skip_header=36,max_rows=1)[1:]
	met_status3 = []
	for i in metastasis3:
		if 'Bone' in i:
			met_status3.append(1)
		else:
			met_status3.append(0)
	# np.savetxt(results+'/GPL570_Y.txt', met_status3) not used

	''' merge GSE14020-GPL96 and GSE14020-GPL570 together '''
	data2Final = data_scaled2.T
	data3Final = data_scaled3.T
	dataXFinal = np.concatenate((data2Final, data3Final))
	np.savetxt(results+'/testXFinal.txt', dataXFinal)
	dataYFinal = np.concatenate((met_status2, met_status3))
	np.savetxt(results+'/testYFinal.txt', dataYFinal)


''' Partitioning method 1 '''
def distinctPartion(X,Y):
	# Calculate the number of bone metastasis samples
	num_true = int(np.sum(Y))
	num_genes = X.shape[0]
	met_true = np.zeros((num_genes,num_true))
	met_false = np.zeros((num_genes,len(Y)-num_true))

	true_counter = 0
	false_counter = 0

	for i in range(0,len(Y)):
		if Y[i] == 1:
			met_true[:,true_counter] = X[:,i]
			true_counter += 1
		else:
			met_false[:,false_counter] = X[:,i]
			false_counter += 1
	
	#partion met_true and met_false by 70%
	split_true = int(met_true.shape[1]*0.7) #69*0.7=48
	split_false = int(met_false.shape[1]*0.7) #217*0.7 = 151
	train_size = split_false+split_true
	#for test file
	split_true_test = int(met_true.shape[1] - split_true) 
	split_false_test = int(met_false.shape[1] - split_false)

	train_x = np.zeros((train_size, num_genes))
	test_x = np.zeros((X.shape[1] - train_size, num_genes))

	met_true_trans = met_true.T 
	met_false_trans = met_false.T

	train_x[:split_true, :] = met_true_trans[:split_true, :]
	train_x[split_true:, :] = met_false_trans[:split_false, :]
	
	test_x[:split_true_test, :] = met_true_trans[split_true:, :]
	test_x[split_true_test:, :] = met_false_trans[split_false:, :]

	train_y = [1]*split_true + [0]*split_false
	test_y = [1]*split_true_test + [0]*split_false_test

	
	return train_x, test_x, train_y, test_y


''' Partitioning method 2 '''
def conventionalPartition(X,Y):
	#random partition by 70:30
	(samples, genes) = X.T.shape
	train_samples = int(samples*0.7)
	train_x = X.T[:train_samples, :]
	train_y = Y[:train_samples]

	test_x = X.T[train_samples:, :]
	test_y = Y[train_samples:]

	return train_x, test_x, train_y, test_y

if __name__ == '__main__':
	main()
