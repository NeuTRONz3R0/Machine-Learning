import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics



ires = pd.read_csv(r"E:\\code stuff\\machine learning\\datasets\\iris.csv")

print(ires.head)
print(ires["flower_type"].value_counts())


ires.plot(kind="scatter", x="sepallength", y="sepalwidth")
sns.FacetGrid(ires, hue="flower_type", palette="husl", size=5) \
   .map(plt.scatter, "sepallength","sepalwidth" ) \
   .add_legend()
sns.boxplot(x="flower_type", y="petallength", palette="husl", data=ires)
plt.show()


train, test = train_test_split(ires, test_size = 0.4, stratify = ires["flower_type"], random_state = 42)

x_train = train[['sepallength','sepalwidth','petallength','petalwidth']]
y_train = train.flower_type
x_test = test[['sepallength','sepalwidth','petallength','petalwidth']]
y_test = test.flower_type

cl = svm.SVC()
cl.fit(x_train,y_train)

y_pred = cl.predict(x_test)
acc = metrics.accuracy_score(y_test,y_pred)*100
print("accuracy using svm-",acc)