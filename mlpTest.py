# Heather Han (hhan16)
# Jiayao Wu (jwu86)
# 29 April 2017

# python version: 2.7

import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
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
	results = {}

	print "********************* WITHOUT FEATURE SELECTION **********************"
	print "\n********** Distinct Partition without Feature Selection: **********"
	result = mlp(distinct_train_x, distinct_test_x, distinct_train_y, distinct_test_y, resultsFolder+'/distinct/')
	results["distinct"] = result

	print "\n********** Conventional Partition without Feature Selection: **********"
	result = mlp(conv_train_x, conv_test_x, conv_train_y, conv_test_y, resultsFolder+'/conventional/')
	results["conventional"] = result

	''' Feature Selection '''
	print "\n********************* WITH FEATURE SELECTION **********************"
	print "\n*********** Distinct Partition with Feature Selection: **********"
	distinct_train_new = np.genfromtxt(sys.argv[10],delimiter=' ')
	distinct_test_new = np.genfromtxt(sys.argv[11],delimiter=' ')
	result = mlp(distinct_train_new, distinct_test_new, distinct_train_y, distinct_test_y, resultsFolder+'/feature_distinct/')
	results["distinctF"] = result

	print "\n********** Conventional Partition with Feature Selection: ***********"
	conv_train_new = np.genfromtxt(sys.argv[12],delimiter=' ')
	conv_test_new = np.genfromtxt(sys.argv[13],delimiter=' ')
	result = mlp(conv_train_new, conv_test_new, conv_train_y, conv_test_y, resultsFolder+'/feature_conventional/')
	results["conventionalF"] = result

	''' plot the results '''
	for i in results:
		count = 1
		for j in results[i]:
			if i == 'distinct':
				plt.scatter(count, j, c = 'navy', marker='*', alpha=0.5, s=60, label='Distinct')
			if i == 'conventional':
				plt.scatter(count, j, c = 'g', marker='^', alpha=0.5, s=60, label='Conventional')
			if i == 'distinctF':
				plt.scatter(count, j, c = 'r', marker='p', alpha=0.5, s=60, label="Feature Select -- Distinct")
			if i == 'conventionalF':
				plt.scatter(count, j, c = 'purple', marker='o', alpha=0.5, s=60, label="Feature Select -- Conventional")
			count += 1

	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys())

	xticks = [' ', 'RELU', 'logistic', ' ']
	plt.xticks([1,2,3,4],xticks)
	plt.title("MLP Classification Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Method")
	plt.show()


def mlp(train_x, test_x, train_y, test_y, resultsFolder):
	''' Generate a MLP for each model '''
	relu_clf = pickle.load(open(resultsFolder+'relu.txt', 'rb'))

	relu_accuracy = relu_clf.score(train_x, train_y)
	relu_accuracy_test = relu_clf.score(test_x, test_y)
	relu_cv = cross_val_score(relu_clf, train_x, train_y, cv=10).mean()
	
	# Gaussian (RBF) kernel SVM
	log_clf = pickle.load(open(resultsFolder+'log.txt', 'rb'))

	log_accuracy = log_clf.score(train_x, train_y)
	log_accuracy_test = log_clf.score(test_x, test_y)
	log_cv = cross_val_score(log_clf, train_x, train_y, cv=10).mean()


	''' Accuracy without cross validation '''
	print "Accuracy without Cross Validation: "
	print "relu activation accuracy: ", relu_accuracy
	print "logistic activation accuracy: ", log_accuracy

	print "\nAccuracy without Cross Validation for testing files: "
	print "relu activation accuracy: ", relu_accuracy_test
	print "logistic activation accuracy: ", log_accuracy_test
	
	''' Accuracy with 5-fold cross validation '''
	print "\nAccuracy with 10-fold Cross Validation: "
	print "relu activation accuracy: ", relu_cv
	print "logistic activation accuracy: ", log_cv

	return relu_cv, log_cv


if __name__ == '__main__':
	main()
