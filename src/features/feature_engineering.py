import numpy as np
import pandas as pd

import os

from sklearn.feature_extraction.text import TfidfVectorizer

# fetch the data from data/processed
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

train_data.fillna('',inplace=True)
test_data.fillna('',inplace=True)

# apply BoW
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply tfidfvectorizer 
vectorizer = TfidfVectorizer(max_features=50)

# Fit the vectorizer on the training data and transform it
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_tfidf.toarray())

train_df['label'] = y_train

test_df = pd.DataFrame(X_test_tfidf.toarray())

test_df['label'] = y_test

# store the data inside data/features
data_path = os.path.join("data","features")

os.makedirs(data_path)

train_df.to_csv(os.path.join(data_path,"train_tfidf.csv"))
test_df.to_csv(os.path.join(data_path,"test_tfidf.csv"))