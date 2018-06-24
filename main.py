import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

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

svm_classifier = svm.SVC(gamma=0.001, C=100)
svm_classifier.fit(x_train, y_train)

svm_pred = svm_classifier.predict(x_test)

print accuracy_score(y_test, svm_pred)

#NN

num_neurons = 100
optimizer = optimizers.SGD(lr=1)
epochs = 50
batch_size = 10

model = Sequential() 
model.add(Dense(8, input_shape=(8,), activation='sigmoid'))
model.add(Dense(num_neurons, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_data, Y_data, epochs = epochs, batch_size = batch_size, verbose = 1)