import os
import pandas as pd
from collections import defaultdict
import gensim
from sklearn import naive_bayes as nb
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier as mlpc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from numpy import mean
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords


data = pd.read_csv('train.csv', header =0)
stop = set(stopwords.words('english'))
ques = []
target = []
for i in range(len(data)):
    line1 = [w.lower() for w in word_tokenize(str(data.iloc[i,3]))]
    line2 = [w.lower() for w in word_tokenize(str(data.iloc[i,4]))]
    line1 = [word for word in line1 if word not in stop]
    line2 = [word for word in line2 if word not in stop]
    ques.append(line1+line2)
    target.append(data.iloc[i,5])

frequency = defaultdict(int)
for que in ques:
    for word in que:
        frequency[word] += 1
ques = [[word for word in que if frequency[word] > 1] for que in ques]

dictionary = gensim.corpora.Dictionary(ques)
n_unique_tokens = len(dictionary)

bag_of_words = [dictionary.doc2bow(que) for que in ques]

dense_bow = gensim.matutils.corpus2dense(bag_of_words, num_terms = n_unique_tokens).transpose()
#tfidf
tfidf = gensim.models.TfidfModel(bag_of_words)
records = tfidf[bag_of_words]
dense_tfidf = gensim.matutils.corpus2dense(records, num_terms = n_unique_tokens).transpose()

#assign the feature set that you want to use for training and classification to dataset
dataset = dense_tfidf

kf = KFold(n_splits = 10, shuffle = True)

accuracies = []
scores = []

for train, test in kf.split(dataset):
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []
    for i in train:
        train_set.append(dataset[i])
        train_labels.append(target[i])
    for i in test:
        test_set.append(dataset[i])
        test_labels.append(target[i])
    classifier = nb.GaussianNB()
    classifier = mlpc(solver = 'lbfgs', hidden_layer_sizes = (5, 15), max_iter = 200)
    predicted = classifier.fit(train_set, train_labels).predict(test_set)
    score = f1_score(test_labels, predicted, average = 'weighted')
    scores.append(score)
    incorrect = (test_labels != predicted).sum()
    accuracy = (len(test_set) - incorrect) / len(test_set) * 100.
    accuracies.append(accuracy)
print("Maximum accuracy attained ", max(accuracies))
print("f1score  ", scores[np.argmax(accuracies)])


