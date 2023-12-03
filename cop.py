import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,r2_score,mean_absolute_error,explained_variance_score
data=pd.read_csv('C:\\Users\\2003v\\python ml\\intern1\\Co-19 TwitDataset (Apr-Jun 2020).csv')
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data.drop(columns=['id'],inplace=True)
print(data.head())
dt=pd.DataFrame()
dt['text']=data['clean_tweet']
dt["sentiment"]=data["sentiment"]
print(data.info())
print(dt.head())
x=dt['text']
y=dt["sentiment"]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=2)
from nltk.tokenize import RegexpTokenizer
# NLTK -> Tokenize -> RegexpTokenizer
# Stemming
# "Playing" -> "Play"
# "Working" -> "Work"
from nltk.stem.porter import PorterStemmer
# NLTK -> Stem -> Porter -> PorterStemmer
from nltk.corpus import stopwords
# NLTK -> Corpus -> stopwords
# Downloading the stopwords
import nltk
nltk.download('stopwords')
tokenizer = RegexpTokenizer(r"\w+")
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()
def getCleanedText(text):
    text = text.lower()
  # tokenizing
    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(tokens) for tokens in new_tokens]
    clean_text = " ".join(stemmed_tokens)
    return clean_text
X_test = ["lockdown was stressfull"]
X_clean = [getCleanedText(i) for i in x_train]
xt_clean = [getCleanedText(i) for i in X_test]
xtest_clean=[getCleanedText(i) for i in x_test]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range = (1,2))
# "I am PyDev" -> "i am", "am Pydev"
X_vec = cv.fit_transform(X_clean).toarray()
print(cv.get_feature_names_out())
Xt_vect = cv.transform(xt_clean).toarray()
xtest_vect = cv.transform(xtest_clean).toarray()

from sklearn.naive_bayes import MultinomialNB
mn = MultinomialNB()
mn.fit(X_vec, y_train)

y_predict = mn.predict(xtest_vect)
comp=pd.DataFrame(y_test)
comp['pred']=y_predict
print(comp)
accuracyy=accuracy_score(y_test,y_predict)
print(accuracyy)
y_pred = mn.predict(Xt_vect)
print(y_pred)
import pickle
with open('model_mn.pkl', 'wb') as file:
    pickle.dump(mn, file)
with open('count_vectorizer.pkl', 'wb') as file:
    pickle.dump(cv, file)
