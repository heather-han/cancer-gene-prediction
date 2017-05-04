# Heather Han (hhan16)
# Jiayao Wu (jwu86)
# 29 April 2017

''' Testing file to run multi-layer perceptron (MLP) classification on the given
    dataset using the MLP model trained in mlpTrain.py '''

# python version: 2.7

import sys
import numpy as np
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle


def main():
	''' Read data file '''
	distinct_train_x = np.genfromtxt(sys.argv[1],delimiter=' ')
	distinct_test_x = np.genfromtxt(sys.argv[2],delimiter=' ')
	distinct_train_y = np.genfromtxt(sys.argv[3],delimiter=' ')
	distinct_test_y = np.genfromtxt(sys.argv[4],delimiter=' ')
	conv_train_x = np.genfromtxt(sys.argv[5],delimiter=' ')
	conv_test_x = np.genfromtxt(sys.argv[6],delimiter=' ')
	conv_train_y = np.genfromtxt(sys.argv[7],delimiter=' ')
	conv_test_y = np.genfromtxt(sys.argv[8],delimiter=' ')
	resultsFolder = sys.argv[9]

	dist_results = {}
	conv_results = {}

	''' Model alone without feature selection nor dimensionality reduction '''
	print "******************************* SVM **********************************"
	print "**************************** MODEL ALONE *****************************"
	print "\n******************** Distinct Partition alone: *********************"
	result = SVM(distinct_train_x, distinct_test_x, distinct_train_y, distinct_test_y, resultsFolder+'/distinct/')
	dist_results["alone"] = result

	print "\n****************** Conventional Partition alone: *******************"
	result = SVM(conv_train_x, conv_test_x, conv_train_y, conv_test_y, resultsFolder+'/conventional/')
	conv_results["alone"] = result

	''' Feature Selection '''
	print "\n********************* WITH FEATURE SELECTION ***********************"
	print "\n************ Distinct Partition with Feature Selection: ************"
	distinct_train_new = np.genfromtxt(sys.argv[10],delimiter=' ')
	distinct_test_new = np.genfromtxt(sys.argv[11],delimiter=' ')
	result = SVM(distinct_train_new, distinct_test_new, distinct_train_y, distinct_test_y, resultsFolder+'/feature_distinct/')
	dist_results["feature"] = result

	print "\n********** Conventional Partition with Feature Selection: **********"
	conv_train_new = np.genfromtxt(sys.argv[12],delimiter=' ')
	conv_test_new = np.genfromtxt(sys.argv[13],delimiter=' ')

	result = SVM(conv_train_new, conv_test_new, conv_train_y, conv_test_y, resultsFolder+'/feature_conventional/')
	conv_results["feature"] = result

	''' Dimensionality Reduction '''
	print "\n****************** WITH DIMENSIONALITY REDUCTION *******************"
	print "\n********* Distinct Partition with Dimensionality Reduction: ********"
	distinct_train_new = np.genfromtxt(sys.argv[14],delimiter=' ')
	distinct_test_new = np.genfromtxt(sys.argv[15],delimiter=' ')
	result = SVM(distinct_train_new, distinct_test_new, distinct_train_y, distinct_test_y, resultsFolder+'/pca_distinct/')
	dist_results["PCA"] = result

	print "\n******* Conventional Partition with Dimensionality Reduction: ******"
	conv_train_new = np.genfromtxt(sys.argv[16],delimiter=' ')
	conv_test_new = np.genfromtxt(sys.argv[17],delimiter=' ')

	result = SVM(conv_train_new, conv_test_new, conv_train_y, conv_test_y, resultsFolder+'/pca_conventional/')
	conv_results["PCA"] = result

	''' for additional dataset '''
	print "\n*********************** Additional Dataset *************************"
	print "\n*********** Feature Selection + conventional partition *************"

	testNew_X = np.genfromtxt(sys.argv[18],delimiter=' ')
	testNew_Y = np.genfromtxt(sys.argv[19],delimiter=' ')
	result = SVM2(testNew_X, testNew_Y, resultsFolder+'/feature_conventional/' )
	conv_results["add_feature"] = result

	print "\n******************* PCA + conventional partition *******************"
	testNew_X = np.genfromtxt(sys.argv[20],delimiter=' ')
	result = SVM2(testNew_X, testNew_Y, resultsFolder+'/pca_conventional/' )
	conv_results["add_pca"] = result

	''' Plot the model accuracies '''
	plot(dist_results, "Distinct")
	plot(conv_results, "Conventional")


def SVM(train_x, test_x, train_y, test_y, resultsFolder):
	''' Generate a SVM for each model '''
	# linear kernel SVM
	lin_svm = pickle.load(open(resultsFolder+'linear.txt', 'rb'))
	lin_accuracy = lin_svm.score(train_x, train_y)
	lin_accuracy_test = lin_svm.score(test_x, test_y)
	lin_cv = cross_val_score(lin_svm, train_x, train_y, cv=10).mean()
	
	# Gaussian (RBF) kernel SVM
	gau_svm = pickle.load(open(resultsFolder+'gaussian.txt', 'rb'))
	gau_accuracy = gau_svm.score(train_x, train_y)
	gau_accuracy_test = gau_svm.score(test_x, test_y)
	gau_cv = cross_val_score(gau_svm, train_x, train_y, cv=10).mean()

	# Polynomial kernel SVM with d = 3
	pol3_svm = pickle.load(open(resultsFolder+'pol3.txt', 'rb'))
	pol3_accuracy = pol3_svm.score(train_x, train_y)
	pol3_accuracy_test = pol3_svm.score(test_x, test_y)
	pol3_cv = cross_val_score(pol3_svm, train_x, train_y, cv=10).mean()

	# Polynomial kernel SVM with d = 4
	pol4_svm = pickle.load(open(resultsFolder+'pol4.txt', 'rb'))
	pol4_accuracy = pol4_svm.score(train_x, train_y)
	pol4_accuracy_test = pol4_svm.score(test_x, test_y)
	pol4_cv = cross_val_score(pol4_svm, train_x, train_y, cv=10).mean()

	# Polynomial kernel SVM with d = 6
	pol6_svm = pickle.load(open(resultsFolder+'pol6.txt', 'rb'))
	pol6_accuracy = pol6_svm.score(train_x, train_y)
	pol6_accuracy_test = pol6_svm.score(test_x, test_y)
	pol6_cv = cross_val_score(pol6_svm, train_x, train_y, cv=10).mean()

	''' Accuracy of the model on training data without cross validation '''
	print "Accuracy without Cross Validation for training files: "
	print "Linear kernel accuracy: ", lin_accuracy
	print "Gaussian kernel accuracy: ", gau_accuracy
	print "Polynomial accuracy with d=3,4,6: ", pol3_accuracy,pol4_accuracy,pol6_accuracy

	''' Accuracy of the model on testing data without cross validation '''
	print "\nAccuracy without Cross Validation for testing files: "
	print "Linear kernel accuracy: ", lin_accuracy_test
	print "Gaussian kernel accuracy: ", gau_accuracy_test
	print "Polynomial accuracy with d=3,4,6: ", pol3_accuracy_test,pol4_accuracy_test,pol6_accuracy_test
	
	''' Accuracy with 10-fold cross validation '''
	print "\nAccuracy with 10-fold Cross Validation for training files: "
	print "Linear kernel accuracy: ", lin_cv
	print "Gaussian kernel accuracy: ", gau_cv
	print "Polynomial accuracy with d=3,4,6: ", pol3_cv, pol4_cv, pol6_cv

	return lin_cv, gau_cv, pol3_cv, pol4_cv, pol6_cv

''' for the additional dataset '''
def SVM2(test_x, test_y, resultsFolder):
	''' Generate a SVM for each model '''
	# linear kernel SVM
	lin_svm = pickle.load(open(resultsFolder+'linear.txt', 'rb'))
	lin_accuracy_test = lin_svm.score(test_x, test_y)
	lin_cv = cross_val_score(lin_svm, test_x, test_y, cv=10).mean()
	
	# Gaussian (RBF) kernel SVM
	gau_svm = pickle.load(open(resultsFolder+'gaussian.txt', 'rb'))
	gau_accuracy_test = gau_svm.score(test_x, test_y)
	gau_cv = cross_val_score(gau_svm, test_x, test_y, cv=10).mean()

	# Polynomial kernel SVM with d = 3
	pol3_svm = pickle.load(open(resultsFolder+'pol3.txt', 'rb'))
	pol3_accuracy_test = pol3_svm.score(test_x, test_y)
	pol3_cv = cross_val_score(pol3_svm, test_x, test_y, cv=10).mean()

	# Polynomial kernel SVM with d = 4
	pol4_svm = pickle.load(open(resultsFolder+'pol4.txt', 'rb'))
	pol4_accuracy_test = pol4_svm.score(test_x, test_y)
	pol4_cv = cross_val_score(pol4_svm, test_x, test_y, cv=10).mean()

	# Polynomial kernel SVM with d = 6
	pol6_svm = pickle.load(open(resultsFolder+'pol6.txt', 'rb'))
	pol6_accuracy_test = pol6_svm.score(test_x, test_y)
	pol6_cv = cross_val_score(pol6_svm, test_x, test_y, cv=10).mean()

	''' Accuracy of the model on testing data without cross validation '''
	print "\nAccuracy without Cross Validation for testing files: "
	print "Linear kernel accuracy: ", lin_accuracy_test
	print "Gaussian kernel accuracy: ", gau_accuracy_test
	print "Polynomial accuracy with d=3,4,6: ", pol3_accuracy_test,pol4_accuracy_test,pol6_accuracy_test
	
	''' Accuracy with 10-fold cross validation '''
	print "\nAccuracy with 10-fold Cross Validation for testing files: "
	print "Linear kernel accuracy: ", lin_cv
	print "Gaussian kernel accuracy: ", gau_cv
	print "Polynomial accuracy with d=3,4,6: ", pol3_cv, pol4_cv, pol6_cv

	return lin_cv, gau_cv, pol3_cv, pol4_cv, pol6_cv

def plot(results, partition):
	''' Method to plot the accuracies of each model ''' 

	# plot the accuracy of every model
	for i in results:
		count = 1
		for j in results[i]:
			if i == 'alone':
				plt.scatter(count, j, c = 'darkorange', marker='o', alpha=0.7, s=110, label='Model Alone')
			if i == 'feature':
				plt.scatter(count, j, c = 'navy', marker='^', alpha=0.7, s=70, label='Feature Selection')
			if i == 'PCA':
				plt.scatter(count, j, c = 'pink', marker='p', alpha=1, s=110, label="Feature Extraction")
			if i == 'add_feature':
				plt.scatter(count, j, c = 'purple', marker='*', alpha=0.7, s=120, label='Secondary Feature Selection')
			if i == 'add_pca':
				plt.scatter(count, j, c = 'red', marker='s', alpha=0.7, s=110, label='Secondary Feature Extraction')
			count += 1

	# get the handles nad labels to eliminate duplicate labels in the legend
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys())

	xticks = ['linear','gaussian','pol3','pol4','pol6']
	plt.xticks([1,2,3,4,5],xticks)
	plt.title("SVM Classification Accuracy with " + partition + " Partitioning")
	plt.ylabel("Accuracy")
	plt.xlabel("SVM Method")
	plt.show()


if __name__ == '__main__':
	main()
