import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split


data_frame = pd.read_csv(r"E:\\code stuff\\machine learning\\datasets\\Spam.csv")
#print(data_frame)

data_frame['spam']=data_frame['type'].map({'spam':1,'ham':0}).astype(int)
data_frame.head()


def splitter(text):
    return text.split()
data_frame['text']= data_frame['text'].apply(splitter)
#print(data_frame['text'][1])


porter = SnowballStemmer('english',ignore_stopwords=False)
def stemming(text):
    return[porter.stem(i)for i in text]
data_frame['text'] = data_frame['text'].apply(stemming)


#import nltk
#nltk.download('wordnet') upto-date already
limit = WordNetLemmatizer()
def lim_it(text):
    return[limit.lemmatize(i,pos='a')for i in text]
data_frame['text']=data_frame['text'].apply(lim_it)
#print(data_frame)


#import nltk
#nltk.download('stopwords') upto-date already
sw=stopwords.words('english')
def stop(text):
    return[i for i in text if not i in sw]
data_frame['text'] = data_frame['text'].apply(stop)
data_frame['text']= data_frame['text'].apply(' '.join)
data_frame.head()
#print(data_frame)



tf=TfidfVectorizer()
y = data_frame.spam.values
x = tf.fit_transform(data_frame['text'])
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.3,shuffle=False)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
#print(lr.coef_)
y_pr = lr.predict(x_test)
from sklearn.metrics import accuracy_score
log =  accuracy_score(y_pr,y_test)*100
print(tf.get_feature_names())

print("accuracy- ",log)