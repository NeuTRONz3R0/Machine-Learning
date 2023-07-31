import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd 
import seaborn as sns 

iris = pd.read_csv(r"E:\\code stuff\\machine learning\\datasets\\iris.csv")
print(iris.head(2))

f1=iris[iris.flower_type=="Iris-setosa"].plot(kind='scatter',x='sepallength',y='sepalwidth',color='red',Label='setosa')
iris[iris.flower_type=="Iris-versicolor"].plot(kind='scatter',x='sepallength',y='sepalwidth',color='green',Label='versicolor',ax=f1)
iris[iris.flower_type=="Iris-virginica"].plot(kind='scatter',x='sepallength',y='sepalwidth',Label='virginica',ax=f1)
f1.set_xlabel("sepal-length")
f1.set_ylabel("sepal-width")
f1.set_title("sepal length V/S width")
f1=plt.gcf
plt.legend()


f1=iris[iris.flower_type=="Iris-setosa"].plot(kind='scatter',x='petallength',y='petalwidth',color='red',Label='setosa')
iris[iris.flower_type=="Iris-versicolor"].plot(kind='scatter',x='petallength',y='petalwidth',color='green',Label='versicolor',ax=f1)
iris[iris.flower_type=="Iris-virginica"].plot(kind='scatter',x='petallength',y='petalwidth',Label='virginica',ax=f1)
f1.set_xlabel("petal-length")
f1.set_ylabel("petal-width")
f1.set_title("petal length V/S width")
f1=plt.gcf
plt.legend()

#plt.show()

import sklearn 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


print("dimension of whole data-",iris.shape)

train,test = train_test_split(iris,test_size=0.3)
print("dimension of train data",train .shape)
print("dimension of test data",test.shape)

train_x = train[['sepallength','sepalwidth','petallength','petalwidth']]
train_y= train.flower_type

test_x =test[['sepallength','sepalwidth','petallength','petalwidth']]
test_y=test.flower_type 


print("training data",train_x.head(2))
print("testing data",test_x.head(2))

print("training data",train_y.head(2))
print("testing data",test_y.head(2))

print("-----------ACCURACY SCORES-------------")
lr = LogisticRegression()
lr.fit(train_x,train_y)
p = lr.predict(test_x)
acc = accuracy_score(p,test_y)*100
print("LR- ",acc)


kn= KNeighborsClassifier()
kn.fit(train_x,train_y)
pr = kn.predict(test_x)
acc= accuracy_score(pr,test_y)*100
print("kNC- ",acc)

sv= svm.SVC()
sv.fit(train_x,train_y)
pre= sv.predict(test_x)
acc= accuracy_score(pre,test_y)*100
print("svm- ",acc)

dc= DecisionTreeClassifier()
dc.fit(train_x,train_y)
pred= dc.predict(test_x)
acc= accuracy_score(pred,test_y)*100
print("DT- ",acc)









