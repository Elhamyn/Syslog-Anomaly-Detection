import pandas as pd
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

def extract_BoW(syslogs_column):
    syslogs = []
    for line in syslogs_column:    
        syslog = re.sub(r"(?:[0-9a-fA-F]:?){12}", "", line) # remove MAC Addresses
        syslog = re.sub('[^a-zA-Z]', ' ', syslog) # keep letters and spaces
        syslog = syslog.lower() 
        syslog = syslog.split() # split text into words
        syslog = [PorterStemmer().stem(word) for word in syslog if not word in set(stopwords.words('english'))] # PS - keep to root of the words
        syslog = ' '.join(syslog) # merge words back into string
        syslogs.append(syslog) 
    
    stop_words = text.ENGLISH_STOP_WORDS.union({"asa", "fw"}) # remove asa and fw from BoW
    cv = CountVectorizer(max_features = 200, stop_words = stop_words) # consider only 200 most used words
    X_train = cv.fit_transform(syslogs).toarray()
    return X_train

def import_training_dataset(filename):
    
     # Importing the training dataset
    training_dataset = pd.read_csv(filename, delimiter = '\t', quoting = 3, header = None, parse_dates = True, names = ["Syslog", "is_it_Anomaly"])
    #print(training_dataset)
    #print("=================================")
    X_train = training_dataset[training_dataset.columns[0]]
    #print(X_train)
    #print("=================================")
    y_train = training_dataset[training_dataset.columns[1]]
    #print(y_train)
    #print("=================================")

    X_train_vectorized = extract_BoW(X_train)
    #print(X_train_vectorized)
    #print("=================================")
    y_train_vectorized = training_dataset.iloc[:, -1].values
    # print(y_train_vectorized)
    #print("=================================")
    return {"training_dataset": training_dataset, "X_train_vectorized": X_train_vectorized,
            "y_train_vectorized": y_train_vectorized}

training_dataset, X_train_vectorized, y_train_vectorized = import_training_dataset("logs_seen_1000.csv")
print(training_dataset)
print(X_train_vectorized)
print(y_train_vectorized)

def model_LR(X_train_vectorized, y_train_vectorized):    
    from sklearn.linear_model import LogisticRegression
    
    classifier_LR = LogisticRegression(penalty='none', random_state = 0)
    classifier_LR.fit(X_train_vectorized, y_train_vectorized)
    
    return classifier_LR
