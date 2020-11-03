#import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pickle

#load data
data = pd.read_csv("../data/divorce.csv",sep=';')
# print(data)

#data
X = data.drop(['Class'], axis=1)
# print(X)

#class
y = data['Class']
# print(y)
# print(y.value_counts())

#check data null
# print(data.shape)
# print(data.describe())
# print("\n")
# print("Kiem tra xem du lieu co bi thieu (NULL) khong?")
# print(data.isnull().sum())

#build the model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
# print(X_train.head())

#### RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Nhãn thực tế : \n", y_test.ravel())
print("Nhãn dự đoán : \n", y_pred)
print("Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" %metrics.accuracy_score(y_test, y_pred))

confusion = confusion_matrix(y_test, y_pred)
print("Confusion_matrix : \n",confusion)

#kiem tra kFold
for i in range(2,6):
	scores = cross_val_score(model, X, y, cv=i)
	print("Do chinh xac cua mo hinh voi nghi thuc kiem tra %d-fold %.3f" %(i, np.mean(scores)))



pickle.dump(model, open('../divorceModelComplex.pkl','wb'))
