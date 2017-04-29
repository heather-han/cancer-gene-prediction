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
	results = sys.argv[9]

	
	mlp(distinct_train_x, distinct_train_y, 'distinct', results)

	mlp(conv_train_x, conv_train_y, 'conventional', results)

	distinct_train_new = selectFeatures(distinct_train_x,distinct_test_x,distinct_train_y,distinct_test_y, "feature_distinct",results)
	mlp(distinct_train_new, distinct_train_y, "feature_distinct",results)

	
	conv_train_new = selectFeatures(conv_train_x,conv_test_x,conv_train_y,conv_test_y, "feature_conventional", results)
	mlp(conv_train_new, conv_train_y, "feature_conventional", results)


def selectFeatures(train_x, test_x, train_y, test_y, name, results):
	clf = ExtraTreesClassifier()
	clf = clf.fit(train_x, train_y)
	model = SelectFromModel(clf, prefit = True)
	train_new = model.transform(train_x)
	test_new = model.transform(test_x)
	np.savetxt(results+'/params/'+name+'/MLPtestX.txt', test_new)
	np.savetxt(results+'/params/'+name+'/MLPtrainX.txt', train_new)
	return train_new

def mlp(train_x, train_y, name, results):
	''' Generate a MLP for each model '''
	relu_clf = MLPClassifier(hidden_layer_sizes=(100),activation='relu',max_iter=10000).fit(train_x, train_y)
	pickle.dump(relu_clf, open(results+'/params/'+name+'/relu.txt', 'wb'))

	# Gaussian (RBF) kernel SVM
	log_clf = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',max_iter=10000).fit(train_x, train_y)
	pickle.dump(log_clf, open(results+'/params/'+name+'/log.txt', 'wb'))


if __name__ == '__main__':
	main()
