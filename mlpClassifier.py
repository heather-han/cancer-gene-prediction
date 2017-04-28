# Heather Han (hhan16)
# Jiayao Wu (jwu86)
# 27 April 2017

# usage: python 
# python version: 2.7

import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

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
	
	results = {}

	print "Distinct Partition without Feature Selection:"
	result = mlp(distinct_train_x, distinct_test_x, distinct_train_y, distinct_test_y)
	results["distinct"] = result

	print "\nConventional Partition without Feature Selection:"
	result = mlp(conv_train_x, conv_test_x, conv_train_y, conv_test_y)
	results["conventional"] = result

	''' Feature Selection '''
	print "\nDistinct Partition with Feature Selection:"
	distinct_train_new, distinct_test_new = selectFeatures(distinct_train_x,distinct_test_x,distinct_train_y,distinct_test_y)
	result = mlp(distinct_train_new, distinct_test_new, distinct_train_y, distinct_test_y)
	results["distinctF"] = result

	print "\nConventional Partition with Feature Selection:"
	conv_train_new, conv_test_new = selectFeatures(conv_train_x,conv_test_x,conv_train_y,conv_test_y)
	result = mlp(conv_train_new, conv_test_new, conv_train_y, conv_test_y)
	results["conventionalF"] = result

	
	# for i in results:
	# 	count = 1
	# 	for j in results[i]:
	# 		if i == 'distinct':
	# 			color = 'b'
	# 			label = "Distinct"
	# 		if i == 'conventional':
	# 			color = 'y'
	# 			label = "Conventional"
	# 		if i == 'distinctF':
	# 			color = 'r'
	# 			label = "Feature Select -- Distinct"
	# 		if i == 'conventionalF':
	# 			color = 'purple'
	# 			label = "Feature Select -- Conventional"
	# 		plt.scatter(count, j, c = color, label=label)
	# 		count += 1


	# colors = ['navy', 'darkorange', 'red', 'yellow']
	# labels = ["Distinct","Conventional","Feature Select -- Distinct","Feature Select -- Conventional"]

	# for result in results:
	# 	for i, color, labels in zip([1,2,3,4,5], colors, labels):
	# 		plt.scatter(i,result[disease == i, 1],color=color,label=labels)

	# plt.legend(loc='best', scatterpoints=1)
	# plt.show()


def selectFeatures(train_x, test_x, train_y, test_y):
	clf = ExtraTreesClassifier()
	clf = clf.fit(train_x, train_y)
	model = SelectFromModel(clf, prefit = True)
	train_new = model.transform(train_x)
	test_new = model.transform(test_x)
	return train_new, test_new


def mlp(train_x, test_x, train_y, test_y):
	''' Generate a MLP for each model '''
	relu_clf = MLPClassifier(activation='relu',max_iter=10000).fit(train_x, train_y)

	relu_accuracy = relu_clf.score(train_x, train_y)
	relu_accuracy_test = relu_clf.score(test_x, test_y)
	relu_cv = cross_val_score(relu_clf, train_x, train_y, cv=10).mean()
	
	# Gaussian (RBF) kernel SVM
	log_clf = MLPClassifier(activation='logistic',max_iter=10000).fit(train_x, train_y)

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