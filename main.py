import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc
import matplotlib.pyplot as plt

# k in cross validation k fold
cv_k = 10

def NN(X_data, Y_data, x_train, x_test, y_train, y_test):
	num_neurons = 1600
	optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
	epochs = 10
	batch_size = 20

	model = Sequential() 
	model.add(Dense(8, input_shape=(8,), activation='sigmoid'))
	model.add(Dense(880, activation='sigmoid'))
	model.add(Dense(540, activation='sigmoid'))
	model.add(Dense(180, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))
	model.summary()
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 0)
	nn_pred = model.predict_classes(x_test)
	print "\n#### NN\n"
	print (accuracy_score(y_test, nn_pred))
	print confusion_matrix(y_test, nn_pred)
	print classification_report(y_test,nn_pred)



def SVM(X_data, Y_data, x_train, x_test, y_train, y_test):
	svm_classifier = svm.SVC(kernel='rbf',C=200, degree=5 ,gamma=0.0001)
	svm_classifier.fit(x_train, y_train)

	svm_pred = svm_classifier.predict(x_test)
	print "\n#### SVM\n"
	print accuracy_score(y_test, svm_pred)
	print confusion_matrix(y_test, svm_pred)

	score =  cross_val_score(svm_classifier, X_data, Y_data, cv=cv_k)
	print sum(score)/len(score)
	print classification_report(y_test,svm_pred)
	
	


def randomForest(X_data, Y_data, x_train, x_test, y_train, y_test):
	rfc = RandomForestClassifier(n_estimators=16)
	rfc.fit(x_train, y_train)
	rfc_pred  = rfc.predict(x_test)
	print "\n#### RF\n"
	print (accuracy_score(y_test, rfc_pred))
	print confusion_matrix(y_test, rfc_pred)
	score =  cross_val_score(rfc, X_data, Y_data, cv=cv_k)
	print sum(score)/len(score)

	print classification_report(y_test,rfc_pred)
	predicting_probabilites = rfc.predict_proba(x_test)[:,1]
	fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
	plt.subplot(222)
	plt.figure(figsize=(12,6))
	plt.plot(fpr,tpr,label = ("Area sob a curva:",auc(fpr,tpr)),color = "r")
	plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
	plt.legend(loc = "best")
	plt.title("Curva ROC - RFC",fontsize=20)
	plt.show()

def naiveBayes(X_data, Y_data, x_train, x_test, y_train, y_test):
	nb = GaussianNB()
	nb_pred = nb.fit(x_train, y_train).predict(x_test)
	print "\n#### NB\n"
	print (accuracy_score(y_test, nb_pred))
	print confusion_matrix(y_test, nb_pred)
	score =  cross_val_score(nb, X_data, Y_data, cv=cv_k)
	print sum(score)/len(score)

	print classification_report(y_test,nb_pred)
	predicting_probabilites = nb.predict_proba(x_test)[:,1]
	fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
	plt.subplot(222)
	plt.figure(figsize=(12,6))
	plt.plot(fpr,tpr,label = ("Area sob a curva:",auc(fpr,tpr)),color = "r")
	plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
	plt.legend(loc = "best")
	plt.title("Curva ROC - NB",fontsize=20)
	plt.show()


def adaClassifier(X_data, Y_data, x_train, x_test, y_train, y_test):
	ada =   AdaBoostClassifier(n_estimators=100, learning_rate=0.25)
	ada.fit(x_train, y_train)
	ada_pred = ada.predict(x_test)
	print "\n#### ADA\n"
	print (accuracy_score(y_test, ada_pred))
	print confusion_matrix(y_test, ada_pred)
	score =  cross_val_score(ada, X_data, Y_data, cv=cv_k)
	print sum(score)/len(score)

	print classification_report(y_test,ada_pred)
	predicting_probabilites = ada.predict_proba(x_test)[:,1]
	fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
	plt.subplot(222)
	plt.figure(figsize=(12,6))
	plt.plot(fpr,tpr,label = ("Area sob a curva:",auc(fpr,tpr)),color = "r")
	plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
	plt.legend(loc = "best")
	plt.title("Curva ROC - Ada",fontsize=20)
	plt.show()


def baggingClassifier(X_data, Y_data, x_train, x_test, y_train, y_test):
	bag = BaggingClassifier(n_estimators = 20)
	bag.fit(x_train, y_train)
	bag_pred = bag.predict(x_test)
	print "\n#### BAG \n"
	print (accuracy_score(y_test, bag_pred))
	print confusion_matrix(y_test, bag_pred)
	score =  cross_val_score(bag, X_data, Y_data, cv=cv_k)
	print sum(score)/len(score)

	print classification_report(y_test,bag_pred)
	predicting_probabilites = bag.predict_proba(x_test)[:,1]
	fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
	plt.subplot(222)
	plt.figure(figsize=(12,6))
	plt.plot(fpr,tpr,label = ("Area sob a curva:",auc(fpr,tpr)),color = "r")
	plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
	plt.legend(loc = "best")
	plt.title("Curva ROC - BAG",fontsize=20)
	plt.show()

def gradientBostingClassifier(X_data, Y_data, x_train, x_test, y_train, y_test):
	grd = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05)
	grd.fit(x_train, y_train)
	grd_pred = grd.predict(x_test)
	print "\n#### GRB \n"
	print (accuracy_score(y_test, grd_pred))
	print confusion_matrix(y_test, grd_pred)
	score =  cross_val_score(grd, X_data, Y_data, cv=cv_k)
	print sum(score)/len(score)

	print classification_report(y_test,grd_pred)
	predicting_probabilites = grd.predict_proba(x_test)[:,1]
	fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
	plt.subplot(222)
	plt.figure(figsize=(12,6))
	plt.plot(fpr,tpr,label = ("Area sob a curva:",auc(fpr,tpr)),color = "r")
	plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
	plt.legend(loc = "best")
	plt.title("Curva ROC - GDB",fontsize=20)
	plt.show()


dataset = pd.read_csv('dataset/pulsar_stars.csv')
#drop missing values
dataset.dropna()
print dataset.head()
print dataset.columns

#only the x_data
X_data = dataset.iloc[:,0:-1].values
print X_data[1]

#Ydata has the labels 0 not a pulsar 1 a pulsar
Y_data = dataset.iloc[:,-1].values
print Y_data[1]

#create test & target partition

x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data)

## DATASET INFO
n_true = 0
for i in range(Y_data.size):
	if(Y_data[i] == 1):
		n_true = n_true + 1
n_false = Y_data.size - n_true
print 'true, false, total on dataset, %'			
print n_true, n_false, Y_data.size, (n_true*100)/Y_data.size

##TRAINING PARTITION INFO
n_true = 0
for i in range(y_train.size):
	if(y_train[i] == 1):
		n_true = n_true + 1

n_false = y_train.size - n_true
print 'true, false, total on training, %'			
print n_true, n_false, y_train.size, (n_true*100)/y_train.size

#TEST PARTITION INFO
n_true = 0
for i in range(y_test.size):
	if(y_test[i] == 1):
		n_true = n_true + 1

n_false = y_test.size - n_true 		
print 'true, false, total on test, % '
print n_true, n_false, y_test.size, (n_true*100)/y_test.size
print "\n\n\n"

#SVM
SVM(X_data, Y_data, x_train, x_test, y_train, y_test)

##Naive bayes
naiveBayes(X_data, Y_data, x_train, x_test, y_train, y_test)

## Random Forest
randomForest(X_data, Y_data, x_train, x_test, y_train, y_test)

##Ada
adaClassifier(X_data, Y_data, x_train, x_test, y_train, y_test)

##Bagging
baggingClassifier(X_data, Y_data, x_train, x_test, y_train, y_test)

##GRD
gradientBostingClassifier(X_data, Y_data, x_train, x_test, y_train, y_test)

#NN
NN(X_data, Y_data, x_train, x_test, y_train, y_test)