# Heather Han (hhan16)
# Jiayao Wu (jwu86)
# 26 April 2017

# usage: python 
# python version: 2.7

import sys
import numpy as np
from sklearn import svm
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
	result = SVM(distinct_train_x, distinct_test_x, distinct_train_y, distinct_test_y)
	results["distinct"] = result

	print "\nConventional Partition without Feature Selection:"
	result = SVM(conv_train_x, conv_test_x, conv_train_y, conv_test_y)
	results["conventional"] = result

	''' Feature Selection '''
	print "\nDistinct Partition with Feature Selection:"
	distinct_train_new, distinct_test_new = selectFeatures(distinct_train_x,distinct_test_x,distinct_train_y,distinct_test_y)
	result = SVM(distinct_train_new, distinct_test_new, distinct_train_y, distinct_test_y)
	results["distinctF"] = result

	print "\nConventional Partition with Feature Selection:"
	conv_train_new, conv_test_new = selectFeatures(conv_train_x,conv_test_x,conv_train_y,conv_test_y)
	result = SVM(conv_train_new, conv_test_new, conv_train_y, conv_test_y)
	results["conventionalF"] = result

	
	for i in results:
		count = 1
		for j in results[i]:
			if i == 'distinct':
				color = 'b'
				label = "Distinct"
			if i == 'conventional':
				color = 'y'
				label = "Conventional"
			if i == 'distinctF':
				color = 'r'
				label = "Feature Select -- Distinct"
			if i == 'conventionalF':
				color = 'purple'
				label = "Feature Select -- Conventional"
			plt.scatter(count, j, c = color, label=label)
			count += 1
	# colors = ['navy', 'darkorange', 'red', 'yellow']
	# labels = ["Distinct","Conventional","Feature Select -- Distinct","Feature Select -- Conventional"]

	# for result in results:
	# 	for i, color, labels in zip([1,2,3,4,5], colors, labels):
	# 		plt.scatter(i,result[disease == i, 1],color=color,label=labels)

	plt.legend(loc='best', scatterpoints=1)
	plt.show()


def selectFeatures(train_x, test_x, train_y, test_y):
	clf = ExtraTreesClassifier()
	clf = clf.fit(train_x, train_y)
	model = SelectFromModel(clf, prefit = True)
	train_new = model.transform(train_x)
	test_new = model.transform(test_x)
	return train_new, test_new


def SVM(train_x, test_x, train_y, test_y):
	''' Generate a SVM for each model '''
	# linear kernel SVM
	lin_svm = svm.SVC(kernel='linear').fit(train_x, train_y)
	lin_accuracy = lin_svm.score(train_x, train_y)
	lin_accuracy_test = lin_svm.score(test_x, test_y)
	lin_cv = cross_val_score(lin_svm, train_x, train_y, cv=10).mean()
	
	# Gaussian (RBF) kernel SVM
	gau_svm = svm.SVC(kernel='rbf').fit(train_x, train_y)
	gau_accuracy = gau_svm.score(train_x, train_y)
	gau_accuracy_test = gau_svm.score(test_x, test_y)
	gau_cv = cross_val_score(gau_svm, train_x, train_y, cv=10).mean()

	# Polynomial kernel SVM with d = 3
	pol3_svm = svm.SVC(kernel='poly',degree=3).fit(train_x, train_y)
	pol3_accuracy = pol3_svm.score(train_x, train_y)
	pol3_accuracy_test = pol3_svm.score(test_x, test_y)
	pol3_cv = cross_val_score(pol3_svm, train_x, train_y, cv=10).mean()

	# Polynomial kernel SVM with d = 4
	pol4_svm = svm.SVC(kernel='poly',degree=4).fit(train_x, train_y)
	pol4_accuracy = pol4_svm.score(train_x, train_y)
	pol4_accuracy_test = pol4_svm.score(test_x, test_y)
	pol4_cv = cross_val_score(pol4_svm, train_x, train_y, cv=10).mean()

	# Polynomial kernel SVM with d = 6
	pol6_svm = svm.SVC(kernel='poly',degree=6).fit(train_x, train_y)
	pol6_accuracy = pol6_svm.score(train_x, train_y)
	pol6_accuracy_test = pol6_svm.score(test_x, test_y)
	pol6_cv = cross_val_score(pol6_svm, train_x, train_y, cv=10).mean()

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
	print "\nAccuracy with 10-fold Cross Validation: "
	print "Linear kernel accuracy: ", lin_cv
	print "Gaussian kernel accuracy: ", gau_cv
	print "Polynomial accuracy with d=3,4,6: ", pol3_cv, pol4_cv, pol6_cv

	return lin_cv, gau_cv, pol3_cv, pol4_cv, pol6_cv


if __name__ == '__main__':
	main()
