import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import traceback
import re
import nltk
import re, string
from tqdm import tqdm
import pickle

import sklearn

sklearn.__version__

dataset = pd.read_csv('IMDB Dataset.csv')

dataset.sentiment.replace('positive', 1, inplace=True)
dataset.sentiment.replace('negative', 0, inplace=True)


dataset_temp = pd.read_csv('new_dataset.csv')

corpus = dataset_temp['review_new']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()

pickle.dump(cv, open('cv-transform.pkl', 'wb'))

y = dataset['sentiment']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

pickle.dump(classifier, open('model_moviewreview.pkl', 'wb'))


   