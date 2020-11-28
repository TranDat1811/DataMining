#import lib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pickle

#load data
data = pd.read_csv("../data/divorce.csv",sep=';')
# print(data)

#data
X = data.drop(['Class'], axis=1)
# print(X)

#class
y = data['Class']
# print(y.value_counts())


#ckeck null data
null = data.isnull().sum()
# print(null)

#sử dụng nghi thức k-fold với k = 170 (leave-one-out)
kf = KFold(n_splits=170)

i=0
tong=0

# #chia tập dữ liệu thành k phần bằng nhau và lặp qua k lần (k=170)
for train_index, test_index in kf.split(X):
	X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
	y_train, y_test = y.iloc[train_index,], y.iloc[test_index]

	print(X_train)
	#xây dựng mô hình
	model = GaussianNB()
	model.fit(X_train,y_train)
	#dự đoán
	thucte = y_test
	dubao = model.predict(X_test)
	i += 1
	#tính độ chính xác của mô hình
	tong += accuracy_score(thucte,dubao)*100
	# print(i,"Độ chính xác :", accuracy_score(thucte,dubao)*100, "\n\tTổng : ",tong)

#tính độ chính xác tổng thể qua k lần lặp
print("Độ chính xác tổng thể trung bình :",tong/170)

# pickle.dump(model, open('../divorceModelComplex.pkl','wb'))


#k = 10
# RandomForest : 97.64
# DecisionTree : 97.05
# GaussianNB : 97.05

#k = 170, Leave-one-out
# RandomForest : 97.64
# DecisionTree : 98.82, 99,41
# GaussianNB : 97.05