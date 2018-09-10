import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# Cleaning the texts
import re # for cleaning text
import nltk
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer

# Importing the dataset
test_dataset = pd.read_csv('test.tsv', delimiter = '\t', quoting = 3)
train_dataset = pd.read_csv('train.tsv', delimiter = '\t', quoting = 3)
word_count=pd.value_counts(train_dataset['Sentiment'].values, sort=False)
print(train_dataset.head())
Index = [1,2,3,4,5]
plt.figure(figsize=(10,5))
plt.bar(Index,word_count,color = 'blue')
plt.xticks(Index,['negative','neutral','somewhat negative','somewhat positive','positive'],rotation=45)
plt.ylabel('word_count')
plt.xlabel('word')
plt.title('Count of Moods')
#plt.bar(Index,len(train_dataset['Sentiment']),color = 'blue')
plt.show()
# stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', "you're", "you've", "you'll",
#              "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
#              'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
#              'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
#              'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
#              'the', 'and', 'but', 'if', 'or', 'because', 'as', 'while', 'of', 'at', 'by', 'for', 'with',
#              'about', 'against', 'between', 'into', 'through', 'during', 'above', 'below', 'to',
#              'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
#              'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'such', 'only', 'own',
#              'same', 'so', 'than', 'too','s',
#              't', 'can', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
#              've', 'y', 'ain', 'aren', "aren't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
#              "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
#              "mustn't", 'needn', "needn't", 'shan', "shan't", 'weren',
#              "weren't", 'won']
def  review_to_words(raw_review):
    review =raw_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))

corpus= []
for i in range(0, 156060):
    corpus.append(review_to_words(train_dataset['Phrase'][i]))

#
# wordcloud = WordCloud(stopwords=STOPWORDS,
#                       background_color='black',
#                       width=3000,
#                       height=2500
#                      ).generate(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = train_dataset.iloc[:, 1].values # all rows from column 1

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
cm = classification_report(y_test, y_pred)
c = confusion_matrix(y_test, y_pred)

print(cm)
print(c)