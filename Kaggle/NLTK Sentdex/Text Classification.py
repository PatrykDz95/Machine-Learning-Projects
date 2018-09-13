import nltk
import random
from nltk.corpus import movie_reviews
import pickle
#nltk.download('movie_reviews')

# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)

#print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words) # from most common words to the least common
word_features = list(all_words.keys())[:3000] # first 3000 most common words

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) # calls True if a word from word_features is in this set of document  // (w in words) creates a boolien: true or false

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

#print(featuresets)

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Bayes algorithm:
# posterior - prior occurences X liklihood / evidence
#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Naive Bayes Algorithm accuracy precent: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15) # shows most popular words on both sides and if they are positive or negative (pos and neg)

# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()






