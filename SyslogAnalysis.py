
import pandas as pd
import numpy
import re
import string
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import argparse
import sys
import numpy as np
from joblib import dump, load
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from keras.models import load_model
ps = nltk.PorterStemmer()
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

training_dataset = pd.read_csv("logs_seen_1000.csv", delimiter = '\t', quoting = 3, header = None, parse_dates = True, names = ["Syslog", "is_it_Anomaly"])
training_dataset.columns = ['X_train', 'y_train']
#X_train = training_dataset[training_dataset.columns[0]]
# print(X_train)
# print("=================================")    
#y_train = training_dataset[training_dataset.columns[1]]
# print(y_train)
# print("=================================")
# print("Training data has {} rows and {} columns".format(len(training_dataset), len(training_dataset.columns)))

testing_dataset = pd.read_csv("logs_unseen_1000.csv", delimiter = '\t', quoting = 3, header = None, parse_dates = True, names = ["Syslog", "is_it_Anomaly"])
testing_dataset.columns = ['X_test', 'y_test']
#X_test = testing_dataset[testing_dataset.columns[0]]
# print(X_test)
# print("=================================")
#y_test = testing_dataset[testing_dataset.columns[1]]
# print(y_test)
# print("=================================")
# print("Testing data has {} rows and {} columns".format(len(testing_dataset), len(testing_dataset.columns)))


stopwords = nltk.corpus.stopwords.words('english')

#clean training data
def clean_training_text(training_text):
    #remove punctuations
    training_text = "".join([word.lower() for word in training_text if word not in string.punctuation])
    #tokenizing
    # \W+ search for any non-word characters
    tokens = re.split('\W+', training_text)
    #Stemm and remove stop words
    training_text = [ps.stem(word) for word in tokens if word not in stopwords]
    return training_text

#vectorizing with count vectorizing method
count_vect_training = CountVectorizer(max_features = 200, analyzer=clean_training_text)
#toarray function is used to transfer the feature vector into two-dimensional array
X_train_vect = count_vect_training.fit_transform(training_dataset['X_train']).toarray()
#print(count_vect_training.get_feature_names())
y_train_vect = training_dataset.iloc[:, -1].values
#print(y_train_vect)
    

#clean testing data
def clean_testing_text(testing_text):
    #remove punctuations
    testing_text = "".join([word.lower() for word in testing_text if word not in string.punctuation])
    #tokenizing
    # \W+ search for any non-word characters
    tokens = re.split('\W+', testing_text)
    #Stemm and remove stop words
    testing_text = [ps.stem(word) for word in tokens if word not in stopwords]
    return testing_text

#vectorizing with count vectorizing method
count_vect_testing = CountVectorizer(max_features = 200, analyzer=clean_testing_text)
#toarray function is used to transfer the feature vector into two-dimensional array
X_test_vect = count_vect_testing.fit_transform(testing_dataset['X_test']).toarray()
#print(count_vect_testing.get_feature_names())
y_test_vect = testing_dataset.iloc[:, -1].values


#KNN model
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'hamming')
classifier_KNN_model = classifier_KNN.fit(X_train_vect, y_train_vect)
y_pred = classifier_KNN_model.predict(X_test_vect)
precision, recall, fscore, support = score (y_test_vect, y_pred)
'Precision: {} / Recall: {} / Accuracy: {}'.format(numpy.round(precision, 3), numpy.round(recall, 3), numpy.round((y_pred == y_test_vect).sum() / len (y_pred), 3))
