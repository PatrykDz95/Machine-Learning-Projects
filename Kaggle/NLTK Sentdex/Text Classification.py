import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
#nltk.download('movie_reviews')
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode # that's how we choose who got the most votes
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes)) # the number of assurance of the most popular vote
        conf = choice_votes / len(votes)
        return conf

short_pos = open("positive.txt", "r").read()
short_neg = open("negative.txt", "r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append((r, "pos"))

for r in short_neg.split('\n'):
    documents.append((r, "neg"))

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) # calls True if a word from word_features is in this set of document  // (w in words) creates a boolien: true or false
    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

#positive data example:
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

#negative data example:
training_set = featuresets[100:]
testing_set = featuresets[:100]


# Bayes algorithm:
# posterior - prior occurences X liklihood / evidence

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Original Naive Bayes Algorithm accuracy precent: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15) # shows most popular words on both sides and if they are positive or negative (pos and neg)

# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
MNB_classifier_f = open("MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(MNB_classifier_f)
MNB_classifier_f.close()
# save_MNB_classifier = open("MNB_classifier.pickle", "wb")
# pickle.dump(MNB_classifier, save_MNB_classifier)
# save_MNB_classifier.close()
print("MNB_classifier accuracy precent: ", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print("GaussianNB_classifier accuracy precent: ", (nltk.classify.accuracy(GaussianNB_classifier, testing_set)) * 100)

# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)

BernoulliNB_classifier_f = open("BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(BernoulliNB_classifier_f)
BernoulliNB_classifier_f.close()
# save_BernoulliNB_classifier = open("BernoulliNB_classifier.pickle", "wb")
# pickle.dump(BernoulliNB_classifier, save_BernoulliNB_classifier)
# save_BernoulliNB_classifier.close()
print("BernoulliNB_classifier accuracy precent: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)

LogisticRegression_classifier_f = open("LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_f)
LogisticRegression_classifier_f.close()
# save_LogisticRegression_classifier = open("LogisticRegression_classifier.pickle", "wb")
# pickle.dump(LogisticRegression_classifier, save_LogisticRegression_classifier)
# save_LogisticRegression_classifier.close()
print("LogisticRegression_classifier accuracy precent: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

# SGDClassifier_classifier = SklearnClassifier(SGDClassifier(max_iter=100))
# SGDClassifier_classifier.train(training_set)

SGDClassifier_classifier_f = open("SGDClassifier_classifier.pickle", "rb")
SGDClassifier_classifier = pickle.load(SGDClassifier_classifier_f)
SGDClassifier_classifier_f.close()
# save_SGDClassifier_classifier = open("SGDClassifier_classifier.pickle", "wb")
# pickle.dump(SGDClassifier_classifier, save_SGDClassifier_classifier)
# save_SGDClassifier_classifier.close()
print("SGDClassifierB_classifier accuracy precent: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

# SVC_classifier = SklearnClassifier(SVC(kernel='linear'))
# SVC_classifier.train(training_set)
# print("SVC_classifier accuracy precent: ", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)

# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)

LinearSVC_classifier_f = open("LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(LinearSVC_classifier_f)
LinearSVC_classifier_f.close()
# save_LinearSVC_classifier = open("LinearSVC_classifier.pickle", "wb")
# pickle.dump(LinearSVC_classifier, save_LinearSVC_classifier)
# save_LinearSVC_classifier.close()
print("LinearSVC_classifier accuracy precent: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)

NuSVC_classifier_f = open("NuSVC_classifier.pickle", "rb")
NuSVC_classifier = pickle.load(NuSVC_classifier_f)
NuSVC_classifier_f.close()
# save_NuSVC_classifier = open("NuSVC_classifier.pickle", "wb")
# pickle.dump(NuSVC_classifier, save_NuSVC_classifier)
# save_NuSVC_classifier.close()
print("NuSVC_classifier accuracy precent: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

voted_classifier = VoteClassifier(classifier, LinearSVC_classifier, MNB_classifier,
                                  BernoulliNB_classifier, SGDClassifier_classifier,
                                  NuSVC_classifier, LogisticRegression_classifier)

print("voted_classifier accuracy precent: ", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %: ", voted_classifier.confidence(testing_set[0][0]) * 100)

print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence %: ", voted_classifier.confidence(testing_set[1][0]) * 100)

print("Classification: ", voted_classifier.classify(testing_set[2][0]), "Confidence %: ", voted_classifier.confidence(testing_set[2][0]) * 100)

print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence %: ", voted_classifier.confidence(testing_set[3][0]) * 100)

print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence %: ", voted_classifier.confidence(testing_set[4][0]) * 100)

print("Classification: ", voted_classifier.classify(testing_set[5][0]), "Confidence %: ", voted_classifier.confidence(testing_set[5][0]) * 100)















