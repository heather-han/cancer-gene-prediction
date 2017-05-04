# Heather Han (hhan16)
# Jiayao Wu (jwu86)
# 29 April 2017

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
from sklearn.decomposition import PCA

#path saved: results/params/distinct... /conventional... etc.

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

	''' distinct partition '''
	SVM(distinct_train_x, distinct_train_y, "distinct", results)

	''' conventional partition ''' 
	SVM(conv_train_x, conv_train_y, "conventional",results)

	''' Feature Selection '''
	test_x_add = np.genfromtxt(sys.argv[10],delimiter=' ')

	distinct_train_new = selectFeatures(distinct_train_x,distinct_test_x,distinct_train_y,distinct_test_y, "feature_distinct",results, test_x_add)
	SVM(distinct_train_new, distinct_train_y, "feature_distinct",results)

	conv_train_new = selectFeatures(conv_train_x,conv_test_x,conv_train_y,conv_test_y, "feature_conventional", results, test_x_add)
	SVM(conv_train_new, conv_train_y, "feature_conventional",results)

	''' Dimentionality Reduction '''
	distinct_train_new_pca = dimensionReduction(distinct_train_x, distinct_test_x, "pca_distinct",results)
	SVM(distinct_train_new_pca, distinct_train_y, "pca_distinct",results)

	conv_train_new_pca = dimensionReduction(conv_train_x, conv_test_x, "pca_conventional", results)
	SVM(conv_train_new_pca, conv_train_y, "pca_conventional",results)

def selectFeatures(train_x, test_x, train_y, test_y, name, results, test_x_add):
	# set the random state to 1 so that the results are consistent
	clf = ExtraTreesClassifier(random_state=1)
	clf = clf.fit(train_x, train_y)
	importance = clf.feature_importances_
	a = np.array(importance)
	# select the top 20 importance genes
	selected = np.argpartition(a,-20)[-20:]
	
	np.savetxt(results+'/params/'+name+'/selectedFeatures.txt', selected)

	model = SelectFromModel(clf, prefit = True)
	train_new = model.transform(train_x)
	test_new = model.transform(test_x)
	test_new_add = model.transform(test_x_add)
	np.savetxt(results+'/params/'+name+'/testX.txt', test_new)
	np.savetxt(results+'/params/'+name+'/trainX.txt', train_new)
	np.savetxt(results+'/params/'+name+'/GPL96_570X.txt', test_new_add)

	return train_new

def dimensionReduction(train_x, test_x, name, results):
	pca = PCA(n_components=80, random_state=1) 
	model = pca.fit(train_x)
	train_new = model.transform(train_x)
	test_new  = model.transform(test_x)
	np.savetxt(results+'/params/'+name+'/testX.txt', test_new)
	np.savetxt(results+'/params/'+name+'/trainX.txt', train_new)
	return train_new

def SVM(train_x, train_y, name, results):
	''' Generate a SVM for each model '''
	# linear kernel SVM
	lin_svm = svm.SVC(kernel='linear').fit(train_x, train_y)
	pickle.dump(lin_svm, open(results+'/params/'+name+'/linear.txt', 'wb'))
	
	# Gaussian (RBF) kernel SVM
	gau_svm = svm.SVC(kernel='rbf').fit(train_x, train_y)
	pickle.dump(gau_svm, open(results+'/params/'+name+'/gaussian.txt', 'wb'))

	# Polynomial kernel SVM with d = 3
	pol3_svm = svm.SVC(kernel='poly',degree=3).fit(train_x, train_y)
	pickle.dump(pol3_svm, open(results+'/params/'+name+'/pol3.txt', 'wb'))


	# Polynomial kernel SVM with d = 4
	pol4_svm = svm.SVC(kernel='poly',degree=4).fit(train_x, train_y)
	pickle.dump(pol4_svm, open(results+'/params/'+name+'/pol4.txt', 'wb'))


	# Polynomial kernel SVM with d = 6
	pol6_svm = svm.SVC(kernel='poly',degree=6).fit(train_x, train_y)
	pickle.dump(pol6_svm, open(results+'/params/'+name+'/pol6.txt', 'wb'))


if __name__ == '__main__':
	main()
