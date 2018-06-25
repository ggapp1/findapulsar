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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def NN(x_train, x_test, y_train, y_test):
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
	print accuracy_score(y_test, nn_pred)
	print confusion_matrix(y_test, nn_pred)



def SVM(x_train, x_test, y_train, y_test):
	svm_classifier = svm.SVC(kernel='rbf',C=200, degree=5 ,gamma=0.0001)
	svm_classifier.fit(x_train, y_train)

	svm_pred = svm_classifier.predict(x_test)
	print "\n#### SVM\n"
	print accuracy_score(y_test, svm_pred)
	print confusion_matrix(y_test, svm_pred)

def randomForest(x_train, x_test, y_train, y_test):
	rfc = RandomForestClassifier()
	rfc.fit(x_train, y_train)
	rfc_pred  = rfc.predict(x_test)
	print "\n#### RF\n"
	print accuracy_score(y_test, rfc_pred)
	print confusion_matrix(y_test, rfc_pred)

def naiveBayes(x_train, x_test, y_train, y_test):
	nb = GaussianNB()
	nb_pred = nb.fit(x_train, y_train).predict(x_test)
	print "\n#### NB\n"
	print accuracy_score(y_test, nb_pred)
	print confusion_matrix(y_test, nb_pred)





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

#SVM
SVM(x_train, x_test, y_train, y_test)

#NN
NN(x_train, x_test, y_train, y_test)

## Random Forest
randomForest(x_train, x_test, y_train, y_test)

##Naive bayes
naiveBayes(x_train, x_test, y_train, y_test)