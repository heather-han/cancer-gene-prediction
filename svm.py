# Heather Han (hhan16)
# Jiayao Wu (jwu86)
# 26 April 2017

# usage: python 
# python version: 2.7

import sys
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


def main():
	''' Read data file '''
	data = np.genfromtxt(sys.argv[1],delimiter=' ')
	metastasis = np.genfromtxt(sys.argv[2],delimiter=' ')

	print("Distinct Partition")
	train_x, test_x, train_y, test_y = distinctPartion(data, metastasis)
	SVM(train_x, test_x, train_y, test_y)

	print("\nConventional Partition")
	train_x, test_x, train_y, test_y = conventionalPartition(data, metastasis)
	SVM(train_x, test_x, train_y, test_y)


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


def SVM(train_x, test_x, train_y, test_y):
	''' Generate a SVM for each model '''
	# linear kernel SVM
	lin_svm = svm.SVC(kernel='linear').fit(train_x, train_y)
	lin_accuracy = lin_svm.score(train_x, train_y)
	lin_accuracy_test = lin_svm.score(test_x, test_y)
	lin_cv = cross_val_score(lin_svm, train_x, train_y, cv=5).mean()
	
	# Gaussian (RBF) kernel SVM
	gau_svm = svm.SVC(kernel='rbf').fit(train_x, train_y)
	gau_accuracy = gau_svm.score(train_x, train_y)
	gau_accuracy_test = gau_svm.score(test_x, test_y)
	gau_cv = cross_val_score(gau_svm, train_x, train_y, cv=5).mean()

	# Polynomial kernel SVM with d = 3
	pol3_svm = svm.SVC(kernel='poly',degree=3).fit(train_x, train_y)
	pol3_accuracy = pol3_svm.score(train_x, train_y)
	pol3_accuracy_test = pol3_svm.score(test_x, test_y)
	pol3_cv = cross_val_score(pol3_svm, train_x, train_y, cv=5).mean()

	# Polynomial kernel SVM with d = 4
	pol4_svm = svm.SVC(kernel='poly',degree=4).fit(train_x, train_y)
	pol4_accuracy = pol4_svm.score(train_x, train_y)
	pol4_accuracy_test = pol4_svm.score(test_x, test_y)
	pol4_cv = cross_val_score(pol4_svm, train_x, train_y, cv=5).mean()

	# Polynomial kernel SVM with d = 6
	pol6_svm = svm.SVC(kernel='poly',degree=6).fit(train_x, train_y)
	pol6_accuracy = pol6_svm.score(train_x, train_y)
	pol6_accuracy_test = pol6_svm.score(test_x, test_y)
	pol6_cv = cross_val_score(pol6_svm, train_x, train_y, cv=5).mean()

	''' Accuracy without cross validation '''
	print "Accuracy without Cross Validation: "
	print "Linear kernel accuracy: ", lin_accuracy
	print "Gaussian kernel accuracy: ", gau_accuracy
	print "Polynomial accuracy with d=3,4,6: ", pol3_accuracy,pol4_accuracy,pol6_accuracy

	print "\nAccuracy without Cross Validation for testing files: "
	print "Linear kernel accuracy: ", lin_accuracy_test
	print "Gaussian kernel accuracy: ", gau_accuracy_test
	print "Polynomial accuracy with d=3,4,6: ", pol3_accuracy_test,pol4_accuracy_test,pol6_accuracy_test
	
	''' Accuracy with 5-fold cross validation '''
	print "\nAccuracy with 5-fold Cross Validation: "
	print "Linear kernel accuracy: ", lin_cv
	print "Gaussian kernel accuracy: ", gau_cv
	print "Polynomial accuracy with d=3,4,6: ", pol3_cv, pol4_cv, pol6_cv



if __name__ == '__main__':
	main()