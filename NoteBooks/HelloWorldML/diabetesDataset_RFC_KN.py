import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data_frame = pd.read_csv(r"E:\\code stuff\\machine learning\\datasets\\Diab.csv")
print(data_frame.head())
print("__________________________________________________________________________________________________")
print ('Rows     : ', data_frame.shape[0])
print ('Cols : ', data_frame.shape[1])
print ('\nFeatures- : \n', data_frame.columns.tolist())
print ('\nMissing values :  ', data_frame.isnull().sum().values.sum())
print ('\nUnique values :  \n', data_frame.nunique())

features = ['Pregnancies', 'Glucose', 'blood pressure', 'skin thickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

r, C = 2, 4
f,a = plt.subplots(r, C, figsize=(18,8) )
row, col = 0, 0
for i, feature in enumerate(features):
    if col == C-1:
        row += 1
    col = i % C 
    data_frame[data_frame.Outcome==0][feature].hist(bins=35, color='blue', alpha=0.5, ax=a[row, col]).set_title(feature)
    data_frame[data_frame.Outcome==1][feature].hist(bins=35, color='orange', alpha=0.7, ax=a[row, col])
    
plt.legend(['Non-Diabetic', 'Diabetic'])
f.subplots_adjust(hspace=0.4)
plt.show()


x=data_frame.iloc[:,:-1]
y=data_frame.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=20,test_size=0.4)
print("x_train-",x_train.shape)
print("y_train-",y_train.shape)
print("x_test-",x_test.shape)
print("y_test-",y_test.shape)
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from sklearn.ensemble import RandomForestClassifier
classifier =  RandomForestClassifier(n_estimators=7,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)
predict= classifier.predict(x_test)

from sklearn.metrics import accuracy_score
result = round(accuracy_score(predict,y_test,2))*100
print("ACCURACY_SCORE'random_forest' -",result)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
predict=knn.predict(x_test)
acu=round(accuracy_score(predict,y_test,2))*100
print("ACCURACY_SCORE 'kNN' -",acu)