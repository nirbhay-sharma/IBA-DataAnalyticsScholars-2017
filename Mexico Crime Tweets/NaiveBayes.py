#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:17:48 2017

@author: otto
"""
# set working directory
url = 'https://raw.githubusercontent.com/nirbhay-sharma/IBA-DataAnalyticsScholars-2017/master/Mexico%20Crime%20Tweets/tweets_cdmx.csv'

import nltk, nltk.data
import pandas as pd
import string
import numpy as np

# Try to predict type of crime.

# Load tweet data as mex
mex = pd.read_csv(url, sep = ',', encoding = 'utf-8')

# descriptives
# list columns
list(mex.columns.values)
mex.describe(include = 'all')

# list all types of crime
list(set(mex.loc[:, 'Crime']))

# stemmer
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("spanish")
stemmer = snowball_stemmer

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed    
  
    
# tokenizer
spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle') # breaks up text into spanish sentences
from nltk.tokenize import word_tokenize # breaks up sentences into words

def tokenize(text):
    tokens = [] # words
    sent_tokens = spanish_tokenizer.tokenize(text) # spanish sentences
    for sent in sent_tokens:
        for word in word_tokenize(sent):
            # remove punctuation from tokesns (note this is after tokenization)
            if word not in string.punctuation:
                tokens.append(word)
    stems = stem_tokens(tokens, stemmer)  # now stem tokens (words)
    return stems
  
    
# stopwords - spanish 
from nltk.corpus import stopwords
stop = set(stopwords.words('spanish'))

# build feature data set. 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
        tokenizer = tokenize,
        min_df=1, # words must appear at least once to become features
        analyzer = 'word', # look at single words
        stop_words = stop, # exclude spanish stop words
        lowercase = True,
        smooth_idf = True, # adds 1 for all feature counts. Prevents 0 probablity when new word not observed in training is observed in test
        use_idf=True)      # use tf-idf reweighting (term frequency times inverse document frequency)


# create data subset with tweets in col 0 and crime type in col 1
tweet_crime = mex[['tweet', 'Crime']]
y = tweet_crime['Crime']             # independent variable 
tweets = tweet_crime['tweet']

# transform tweets into tweet-feature matrix
X = vectorizer.fit_transform(tweets) # dependent variable
X.shape
# list feature names
vectorizer.get_feature_names()
# list feature names with counts
vectorizer.vocabulary_

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42)

# Train Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)

# NB prediction 
predicted = clf.predict(X_test)
np.mean(predicted == y_test)

# Train SVM
from sklearn.svm import SVC

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf')
    svm.fit(X, y)
    return svm

# SVM prediciton
svm = train_svm(X_train, y_train)
pred = svm.predict(X_test)

from sklearn.metrics import confusion_matrix
print(svm.score(X_test, y_test))
print(confusion_matrix(pred, y_test))






#for sent in spanish_tokenizer.tokenize('Hola amigo. Estoy bien.'):
#    word_tokenize(sent)
#word_tokenize(spanish_tokenizer.tokenize('Hola amigo. Estoy bien.')[0])     
#
#text_sample = 'Hola amigo. Estoy bien.'
#for ch in text_sample:
#    print(ch)
#
#
#max_count = 0
#max_word = []
#for word, count in vectorizer.vocabulary_.items():
#    if count > max_count:
#        max_word = word
#            
#print(max_word)
