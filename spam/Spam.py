import pandas as pd
import numpy as np
import math
import re

noise = ['!', '?', '.', ',', ':', ';']
with open('./stopwords.txt', 'r') as f:
    sw = [w[-1] for w in f.readlines()]

def make_dict(list1):
    result = []
    for sent in list1:
        words = re.split(r'\W+', sent)
        for word in words:
            if word not in sw:
                result.append(word)
    return result

def fit(dataset, alpha):
    classes, freq, tot = {}, {}, set()
    for feats, label in dataset:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1
        for feat in feats:
            if (label, feat) not in freq:
                freq[(label, feat)] = 0
            freq[(label, feat)] += 1
        tot.add(tuple(feats))

    for label, feat in freq:
        freq[(label, feat)] = (alpha + freq[(label, feat)]) / (alpha*len(tot) + classes[label])
    for c in classes:
        classes[c] /= len(dataset)

    return classes, freq

def classify(classifier, features):
    classes, freq = classifier
    return min(classes.keys(), key=lambda cls: -math.log(classes[cls]) + sum(
        -math.log(freq.get((cls, feat), 10 ** (-7))) for feat in features))

df = pd.read_csv('./spam.csv', encoding='utf8')
m = {'ham': 1, 'spam': 0}
df['v1'] = df['v1'].map(m)

spam = df[df.v1 == 0]['v2'].values
ham = df[df.v1 == 1]['v2'].values

ps = int(len(spam)*.8)
ph = int(len(ham)*.8)
spam_train, spam_test = spam[:ps], spam[ps:]
ham_train, ham_test = ham[:ph], ham[ph:]

dataset = list(map(lambda x: ([x], 0), make_dict(spam_train))) + list(map(lambda x: ([x], 1), make_dict(ham_train)))

classifier = fit(dataset, 0)

from sklearn.externals import joblib
joblib.dump(classifier, './dicts.pkl')

#print(classifier)
print('Spam messsages')
for s in spam_test:
    print(classify(classifier, make_dict([s])), s)
print('Ham messsages')
for h in ham_test:
    print(classify(classifier, make_dict(h)), h)