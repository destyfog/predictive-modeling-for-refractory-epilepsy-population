from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *


import utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import parse_svm_light_data

from sklearn.metrics import roc_curve, auc


def logistic_regression_pred(X_train, Y_train, X_test):
	.logreg = LogisticRegression()
	logreg.fit(X_train, Y_train)
	pre = logreg.predict(X_test)
	return pre


def svm_pred(X_train, Y_train, X_test):
	clf = LinearSVC()
	clf.fit(X_train, Y_train)
	pre=clf.predict(X_test)

	return pre


def decisionTree_pred(X_train, Y_train, X_test):
	clf = DecisionTreeClassifier()
	clf.fit(X_train, Y_train)
	pre=clf.predict(X_test)
	return pre


from sklearn.linear_model import SGDClassifier
def SGDClassifier_pred(X_train, Y_train, X_test):
	clf = SGDClassifier(loss="hinge", penalty="l2")
	clf.fit(X_train, Y_train)
	pre = clf.predict(X_test)
	return pre

from sklearn.ensemble import AdaBoostClassifier
def adaboost(X_train, Y_train, X_test):
	clf = AdaBoostClassifier()
	clf.fit(X_train, Y_train)
	pre = clf.predict(X_test)
	return pre

from sklearn.linear_model import LogisticRegressionCV
def LRCV(X_train, Y_train, X_test):
	clf=LogisticRegressionCV()
	clf.fit(X_train, Y_train)
	pre = clf.predict(X_test)
	return pre

from sklearn.svm import SVC
def dosvc(X_train, Y_train, X_test):
	clf = SVC()
	clf.fit(X_train, Y_train)
	pre = clf.predict(X_test)
	return pre

from sklearn.ensemble import RandomForestClassifier
def RFC(X_train, Y_train, X_test):
	clf = RandomForestClassifier(n_estimators=20, criterion='entropy',max_features=None)
	clf.fit(X_train, Y_train)
	pre = clf.predict(X_test)
	return pre

from sklearn.ensemble import GradientBoostingClassifier
def GBC(X_train, Y_train, X_test):
	clf = GradientBoostingClassifier()
	clf.fit(X_train, Y_train)
	pre = clf.predict(X_test.toarray())
	return pre

def classification_metrics(Y_pred, Y_true):

	acc=accuracy_score(Y_true, Y_pred)
	auc_=roc_auc_score(Y_true, Y_pred)
	precision=precision_score(Y_true, Y_pred)
	recall=recall_score(Y_true, Y_pred)
	f1score=f1_score(Y_true, Y_pred)

	return acc,auc_,precision,recall,f1score

def display_metrics(classifierName,X,Y):
	print "______________________________________________"
	print "Classifier: "+classifierName
	print "kfoldCV"
	acc, auc_, precision, recall, f1score = get_acc_auc_kfold(classifierName,X,Y)
	print "Accuracy: "+str(acc)
	print "AUC: "+str(auc_)
	print "Precision: "+str(precision)
	print "Recall: "+str(recall)
	print "F1-score: "+str(f1score)
	print "______________________________________________"
	print ""
	print "______________________________________________"
	print "Classifier: "+classifierName
	print "randomisedCV"
	acc, auc_, precision, recall, f1score = get_acc_auc_randomisedCV(classifierName,X,Y)
	print "Accuracy: "+str(acc)
	print "AUC: "+str(auc_)
	print "Precision: "+str(precision)
	print "Recall: "+str(recall)
	print "F1-score: "+str(f1score)
	print "______________________________________________"
	print ""


from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean


def get_acc_auc_kfold(clfname,X,Y,k=5):

	acc=[]
	auc_=[]
	precision=[]
	recall=[]
	f1score=[]



	cv = KFold(len(Y), k)
	for train_index, test_index in cv:
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		Y_pred = 0
		if clfname=='Logistic Regression':
			Y_pred=logistic_regression_pred(X_train,Y_train,X_test)
		if clfname=='SVM':
			Y_pred=svm_pred(X_train,Y_train,X_test)
		if clfname=='Decision Tree':
			Y_pred=decisionTree_pred(X_train,Y_train,X_test)
		if clfname=='SGDClassifier':
			Y_pred=SGDClassifier_pred(X_train, Y_train, X_test)
		if clfname=='adaboost':
			Y_pred=adaboost(X_train, Y_train, X_test)
		if clfname=='LogisticRegressionCV':
			Y_pred=LRCV(X_train, Y_train, X_test)
		if clfname=='SVC':
			Y_pred=dosvc(X_train, Y_train, X_test)
		if clfname=='RFC':
			Y_pred=RFC(X_train, Y_train, X_test)
		if clfname=='GBC':
			Y_pred=GBC(X_train, Y_train, X_test)

		accvalue, auc_value, precisionvalue, recallvalue, f1scorevalue = classification_metrics(Y_pred,Y_test)

		acc.append(accvalue)
		auc_.append(auc_value)
		precision.append(precisionvalue)
		recall.append(recallvalue)
		f1score.append(f1scorevalue)



	acc_mean=mean(acc)
	auc_mean=mean(auc_)
	precision_mean = mean(precision)
	recall_mean = mean(recall)
	f1score_mean = mean(f1score)


	return acc_mean,auc_mean,precision_mean,recall_mean,f1score_mean



def get_acc_auc_randomisedCV(clfname,X,Y,iterNo=5,test_percent=0.2):
	acc=[]
	auc_=[]
	precision=[]
	recall=[]
	f1score=[]
	rs = ShuffleSplit(len(Y), iterNo,test_percent)
	for train_index, test_index in rs:

		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		Y_pred = 0
		if clfname=='Logistic Regression':
			Y_pred=logistic_regression_pred(X_train,Y_train,X_test)
		if clfname=='SVM':
			Y_pred=svm_pred(X_train,Y_train,X_test)
		if clfname=='Decision Tree':
			Y_pred=decisionTree_pred(X_train,Y_train,X_test)
		if clfname=='SGDClassifier':
			Y_pred=SGDClassifier_pred(X_train, Y_train, X_test)
		if clfname=='adaboost':
			Y_pred=adaboost(X_train, Y_train, X_test)
		if clfname=='LogisticRegressionCV':
			Y_pred=LRCV(X_train, Y_train, X_test)
		if clfname=='SVC':
			Y_pred=dosvc(X_train, Y_train, X_test)
		if clfname=='RFC':
			Y_pred=RFC(X_train, Y_train, X_test)
		if clfname=='GBC':
			Y_pred=GBC(X_train, Y_train, X_test)
		accvalue, auc_value, precisionvalue, recallvalue, f1scorevalue = classification_metrics(Y_pred,Y_test)

		acc.append(accvalue)
		auc_.append(auc_value)
		precision.append(precisionvalue)
		recall.append(recallvalue)
		f1score.append(f1scorevalue)



	acc_mean=mean(acc)
	auc_mean=mean(auc_)
	precision_mean = mean(precision)
	recall_mean = mean(recall)
	f1score_mean = mean(f1score)


	return acc_mean,auc_mean,precision_mean,recall_mean,f1score_mean

def drawrocdt(X,Y):
	rs = ShuffleSplit(len(Y), 5,0.2)
	i=5

	for train_index, test_index in rs:
		clf = DecisionTreeClassifier()

		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]



		clf.fit(X_train,Y_train)


		pre = clf.predict_proba(X_test)
		#print pre

		y_test_prob = pre[:,1].tolist()
		y_test = Y_test.tolist()

		fpr, tpr, _ = roc_curve(y_test, y_test_prob)


		#print (fpr,tpr)
		roc_auc = auc(fpr, tpr)
		#Plot of a ROC curve for a specific class
		plt.figure()
		plt.plot(fpr, tpr, label='ROC curve')# (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Decision Tree Classifier')
		plt.legend(loc="lower right")
		plt.savefig("pic"+str(i))
		i=i+1


def drawroclr(X,Y):
	rs = ShuffleSplit(len(Y), 5,0.2)
	i=0

	for train_index, test_index in rs:
		clf = LogisticRegression()

		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]



		clf.fit(X_train,Y_train)


		# X_train, Y_train = utils.get_data_from_svmlight("big2/train/part-r-00000")
		# X_test, Y_test = utils.get_data_from_svmlight("big2/test/part-r-00000")

		pre = clf.predict_proba(X_test)


		y_test_prob = pre[:,1]
		y_test = Y_test

		fpr, tpr, _ = roc_curve(y_test, y_test_prob)

		#print (fpr,tpr)
		roc_auc = auc(fpr, tpr)
		#Plot of a ROC curve for a specific class
		plt.figure()
		plt.plot(fpr, tpr, label='ROC curve')#  (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Logistic Regression')
		plt.legend(loc="lower right")
		plt.savefig("pic"+str(i))
		i=i+1

def drawrocada(X,Y):
	rs = ShuffleSplit(len(Y), 5,0.2)
	i=10

	for train_index, test_index in rs:
		clf = AdaBoostClassifier()

		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]



		clf.fit(X_train,Y_train)


		# X_train, Y_train = utils.get_data_from_svmlight("big2/train/part-r-00000")
		# X_test, Y_test = utils.get_data_from_svmlight("big2/test/part-r-00000")

		pre = clf.predict_proba(X_test)


		y_test_prob = pre[:,1]
		y_test = Y_test

		fpr, tpr, _ = roc_curve(y_test, y_test_prob)

		#print (fpr,tpr)
		roc_auc = auc(fpr, tpr)
		#Plot of a ROC curve for a specific class
		plt.figure()
		plt.plot(fpr, tpr, label='ROC curve')#  (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('AdaBoost Classifier')
		plt.legend(loc="lower right")
		plt.savefig("pic"+str(i))
		i=i+1

def drawrocrfc(X,Y):
	rs = ShuffleSplit(len(Y), 5,0.2)
	i=15

	for train_index, test_index in rs:
		clf = RandomForestClassifier(n_estimators=20, criterion='entropy',max_features=None)

		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]



		clf.fit(X_train,Y_train)


		pre = clf.predict_proba(X_test)


		y_test_prob = pre[:,1]
		y_test = Y_test

		fpr, tpr, _ = roc_curve(y_test, y_test_prob)

		#print (fpr,tpr)
		roc_auc = auc(fpr, tpr)
		#Plot of a ROC curve for a specific class
		plt.figure()
		plt.plot(fpr, tpr, label='ROC curve')#  (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Random Forest Classifier')
		plt.legend(loc="lower right")
		plt.savefig("pic"+str(i))
		i=i+1

def main():
	X, Y = utils.get_data_from_svmlight("data/allfeature.data")
	#X, Y = utils.get_data_from_svmlight("data/biggerallfeature.data")

	classifierName = ['Logistic Regression', 'SVM',,'SGDClassifier','adaboost','Decision Tree','RFC','GBC','RFC']
	for clfname in classifierName:
		display_metrics(clfname,X,Y)
	drawrocdt(X,Y)
	drawroclr(X,Y)
	drawrocada(X,Y)
	drawrocrfc(X,Y)



if __name__ == "__main__":
	main()
