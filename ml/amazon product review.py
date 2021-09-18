import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv(r"E:\\code stuff\\machine learning\\datasets\\Amazon.csv")
#print(data.head())

data=data.fillna({'review':''})
def punc_remove(text):
    import string
    return text.translate(text.maketrans('','',string.punctuation))
data['review_clean']=data['review'].apply(punc_remove)
data=data[data['rating']!=3]

data['sentiment']= data['rating'].apply(lambda rating : +1 if rating>3 else -1)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
y = data['sentiment']
x = tf.fit_transform(data['review_clean'])
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.3,shuffle=False)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train,y_train)

pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,y_test)*100
print("accuracy-",acc)