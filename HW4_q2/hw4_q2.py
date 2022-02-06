import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
from xgboost import XGBClassifier
# from sklearn import preprocessing
def convert_cat_to_num(encoder, data):
	encoder.fit(data)
	return encoder.transform(data)

def convert_num_to_cat(encoder, data):
	return encoder.inverse_transform(data)

if __name__ == '__main__':
	testFile = open('adult.test')
	namesFile = open('adult.names')
	trainFile = open('adult.data')

	testData = np.genfromtxt(testFile, delimiter=',', dtype=np.str_)
	trainData = np.genfromtxt(trainFile, delimiter=',', dtype=np.str_)
	#namesData = np.genfromtxt(namesFile, delimiter=',', skip_header=96, dtype=np.str_)

	splitter = trainData.shape[0]
	combinedData = np.concatenate((trainData, testData), axis=0)

	cat_list=[0,1,0,1,0,1,1,1,1,1,0,0,0,1,1]
	enc = []
	for i in range(len(cat_list)):
		enc.append(OrdinalEncoder())

	for i in range(len(cat_list)):
		if (cat_list[i]==1):
			combinedData[:,i:i+1] = convert_cat_to_num(enc[i], combinedData[:,i:i+1])


	X_train = combinedData[:splitter][:,:-1]
	y_train = combinedData[:splitter][:,-1:]

	X_test = combinedData[splitter:][:,:-1]
	y_test = combinedData[splitter:][:,-1:]
	# XX = np.concatenate((X_train, X_test), axis=0)
	# yy = np.concatenate((y_train, y_test), axis=0)

	print(X_test)
	print(y_test)

	print(X_test[0])
	print(trainData[0])
	model = XGBClassifier()
	model.fit(X_train, y_train)



