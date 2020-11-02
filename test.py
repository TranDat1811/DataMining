import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import seaborn as sns

data = pd.read_csv("divorce.csv",sep=';')
print(data)
X = data.drop(['Class'], axis=1)
print(X)
y = data['Class']
print(y)

print(data.describe())
print("\n")
print("Kiem tra xem du lieu co bi thieu (NULL) khong?")
print(data.isnull().sum())

def RandomForest(X,y,test_size):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)
	print(X_train.head())
	model = RandomForestClassifier(n_estimators=100)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	print("Nhãn dự đoán : \n", y_pred)
	print("Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" %metrics.accuracy_score(y_test, y_pred))
	confusion = confusion_matrix(y_test,model.predict(X_test))
	print("Confusion_matrix : \n",confusion)

RandomForest(X,y,test_size=0.4)