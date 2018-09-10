import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re  # for cleaning text
import nltk
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv('Spam.csv', encoding='latin-1')
dataset = dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
dataset = dataset.rename(columns={"v1": "class", "v2": "text"})

print(dataset.head())

dataset['length'] = dataset['text'].apply(len)

print(dataset.head())

# stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', "you're", "you've", "you'll",
#              "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
#              'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
#              'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
#              'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
#              'the', 'and', 'but', 'if', 'or', 'because', 'as', 'while', 'of', 'at', 'and', 'by', 'for', 'with',
#              'about', 'against', 'between', 'into', 'through', 'during', 'above', 'below', 'to',
#              'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
#              'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'such', 'only', 'own',
#              'same', 'so', 'than', 'too', 's',
#              't', 'can', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
#              've', 'y', 'ain', 'aren', "aren't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
#              "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
#              "mustn't", 'needn', "needn't", 'shan', "shan't", 'weren',
#              "weren't", 'won']


def pre_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
        stemmer = SnowballStemmer("english")
        words += (stemmer.stem(i)) + " "
    return words


textFeatures = dataset['text'].copy()
textFeatures = textFeatures.apply(pre_process)
# TFIDF (term frequency - inverse document frequency)
# is a statistical method to tell how important a word
# is to a particular document by increasing the numerical value
# for an occurrence in the specific document but decreasing relative
# to number of occurrences in the entire corpus.
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(textFeatures)
features_train, features_test, labels_train, labels_test = train_test_split(features, dataset['class'], test_size=0.3,
                                                                            random_state=111)

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=0.2)
mnb.fit(features_train, labels_train)
prediction = mnb.predict(features_test)
print(accuracy_score(labels_test,prediction))