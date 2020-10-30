import json
import codecs
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
nltk.download('stopwords')


with open('/Users/chenchiyu/Desktop/COMP90042 NLP/Project/project-files/train.json') as train:
    train_data = json.load(train)

with open('/Users/chenchiyu/Desktop/COMP90042 NLP/Project/project-files/dev.json') as dev1:
    dev = json.load(dev1)

with open('/Users/chenchiyu/Desktop/COMP90042 NLP/Project/project-files/test-unlabelled.json') as test1:
    test = json.load(test1)


def unmangle_utf8(match):
    escaped = match.group(0)                   # '\\u00e2\\u0082\\u00ac'
    hexstr = escaped.replace(r'\u00', '')      # 'e282ac'
    buffer = codecs.decode(hexstr, "hex")      # b'\xe2\x82\xac'

    try:
        return buffer.decode('utf8')           # '€'
    except UnicodeDecodeError:
        print("Could not decode buffer: %s" % buffer)


compose = pd.DataFrame(columns=['label','text'])
compose_dev = pd.DataFrame(columns=['label','text'])


# extract data from json file and convert to dataframe
for event in dev:
    # event : 'train-1'...
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    unUTF = re.sub(r"(?i)(?:\\u00[0-9a-f]{2})+", unmangle_utf8, dev[event]['text'])
    unUTF = unUTF.replace("<|endoftext>|"," ")
    unUTF = unUTF.replace("<|endoftext|>"," ")
    unUTF = unUTF.replace("|<endoftext|>", " ")
    unUTF = re.sub("[^a-zA-z]", ' ', unUTF)
    unUTF = unUTF.lower()
    unUTF = unUTF.split()
    # new_unUTF = []
    # for word in unUTF:
    #     new_unUTF.append(lemmatizer.lemmatize(word))
    # unUTF = new_unUTF
    unUTF = [ps.stem(word) for word in unUTF if not word in set(stopwords.words('english'))]
    unUTF = " ".join(unUTF)
    compose_dev.loc[int(event[4:])] = [dev[event]['label'],unUTF]
    # train-1: [6:] dev-0:[4:]
    # convert train data to dataframe

compose_dev = compose_dev.sort_index(axis=0)


# ========================== For train =================================

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 2,stratify=y)
# svm = OneClassSVM(gamma='scale', nu=0.01)
# X_train = X_train[y_train == 1]
# svm.fit(X_train)
# X_predict = svm.predict(X_test)


# ========================== For dev =================================

cv = CountVectorizer()

dev_X = cv.fit_transform(compose_dev['text']).toarray()
dev_y = compose_dev.iloc[:, 0].values
dev_X_train, dev_X_test, dev_y_train, dev_y_test = train_test_split(dev_X, dev_y, test_size = 0.3, random_state=84)

dev_y_train = dev_y_train.astype(numpy.int)
dev_y_test = dev_y_test.astype(numpy.int)

clfs = [KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),
        MultinomialNB(),LinearSVC(),LogisticRegression(),GaussianNB()]

# for clf in clfs:
#     clf.fit(dev_X_train, dev_y_train)
#     dev_y_pred = clf.predict(dev_X_test)
#     dev_y_pred = dev_y_pred.astype(numpy.int)
#     score = f1_score(dev_y_test, dev_y_pred, pos_label=1)
#     print(clf)
#     print('F1 Score: %.3f'% score)

DT = DecisionTreeClassifier()

# grid search
for C in [0.01,1,100]:
    for gamma in [0.001,0.01,1]:
        rbf_svc = SVC(C=C, gamma=gamma)
        rbf_svc.fit(dev_X_train, dev_y_train)
        dev_y_pred = rbf_svc.predict(dev_X_test)
        dev_y_pred = dev_y_pred.astype(numpy.int)
        print(rbf_svc)
        print(accuracy_score(dev_y_test, dev_y_pred))
        print(f1_score(dev_y_test, dev_y_pred))

for C in [0.0000001,0.0001,1,1000]:
    svc = LinearSVC(C=C )
    svc.fit(dev_X_train, dev_y_train)
    dev_y_pred = svc.predict(dev_X_test)
    dev_y_pred = dev_y_pred.astype(numpy.int)
    print(svc)
    print(accuracy_score(dev_y_test, dev_y_pred))
    print(f1_score(dev_y_test, dev_y_pred))

# ========================== For scoring.py =================================

# creat_dict = pd.DataFrame(index=['label'])
#
# for event in dev:
#     # event : 'dev-1'
#     creat_dict[event] = int(dev_y_pred[int(event[4:])])
#
# result_dev = creat_dict.to_dict()
#
# with open('/Users/chenchiyu/Desktop/COMP90042 NLP/Project/project-files/predict_dev.json', 'w') as f:
#     json.dump(result_dev, f)


# ========================== For test =================================
#
#
# compose_test = pd.DataFrame(columns=['text'])
# svc = LinearSVC(C=100000)
#
# for event in test:
#     # event : 'test-1'
#     ps = PorterStemmer()
#     unUTF = re.sub(r"(?i)(?:\\u00[0-9a-f]{2})+", unmangle_utf8, test[event]['text'])
#     unUTF = unUTF.replace("<|endoftext>|"," ")
#     unUTF = unUTF.replace("<|endoftext|>"," ")
#     unUTF = unUTF.replace("|<endoftext|>", " ")
#     unUTF = re.sub("[^a-zA-z]", ' ', unUTF)
#     unUTF = unUTF.lower()
#     unUTF = unUTF.split()
#     ps = PorterStemmer()
#     unUTF = [ps.stem(word) for word in unUTF if not word in set(stopwords.words('english'))]
#     unUTF = " ".join(unUTF)
#     compose_test.loc[int(event[5:])] = [unUTF]
#
# compose_test = compose_test.sort_index(axis=0)
# '''
#                                                     text
# -1409  sarah palin teleprompt freez iowa saturday for...
# -1408  latest climat report feed alarmist fearmong la...
# -1407  electr car produc less co petrol vehicl studi ...
# -1406  pele diego maradona franz beckenbaeur forev re...
# -1405  warwick ask voter back radic council tax rise ...
# ...                                                  ...
# -4     global warm caus frequent heatwav record break...
# -3     new york keep kid away santa bar goer dress ch...
# -2     turkey could take big step backward human righ...
# -1     washington republican presidenti front runner ...
#  0     market execut somehow unabl stop prevent expla...
# '''
# test_X = cv.transform(compose_test['text']).toarray()
#
# svc.fit(dev_X_train,dev_y_train)
# test_y_pred = svc.predict(test_X)
# test_y_pred = test_y_pred.astype(numpy.int)
#
#
# test_dict = pd.DataFrame(index=['label'])
#
# for event in test:
#     # event : 'test-0'
#     test_dict[event] = int(test_y_pred[int(event[5:])])
#
#
# result_test = test_dict.to_dict()
#
#
# with open('/Users/chenchiyu/Desktop/COMP90042 NLP/Project/project-files/test-output.json', 'w') as f:
#     json.dump(result_test, f)
