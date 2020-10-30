import nltk
nltk.download("reuters") # if necessary
from nltk.corpus import reuters

for category in reuters.categories():
    # reuters.categories len = 90, ['acq', 'alum', 'barley', 'bop', 'carcass'...]
    # category = 'acq'...
    print (category, len(reuters.fileids(category)))
    # reuters.fileids, len = 10,788 ['test/14826', 'test/14828', 'test/14829'...,'train/34322']
    # reuters.fileids(category) 根据category找出符合的fileids，['test/***',...]


from sklearn.feature_extraction import DictVectorizer


def get_BOW(text):
    # text = ['word1','word2',...]
    # BOW={'acq':3,'bad':2,...}
    BOW = {}
    for word in text:
        BOW[word] = BOW.get(word,0) + 1
    return BOW


def prepare_reuters_data(topic,feature_extractor):
    training_set = []
    training_classifications = []
    test_set = []
    test_classifications = []
    for file_id in reuters.fileids():
        # file_id 每一个file
        feature_dict = feature_extractor(reuters.words(file_id))
        # feature_extractor = get_BOW
        # reuters.words('testid') 此news中所有的words, 有重复
        # feature_dict 建立此file的字典

        if file_id.startswith("train"):
            # 开头为train的file
            training_set.append(feature_dict)
            if topic in reuters.categories(file_id):
                # reuters.categories('file id') 获取该file的category
                # 比较是否为设定的topic
                training_classifications.append(topic)
            else:
                training_classifications.append("not " + topic)
        else:
            # 开头为test的file
            test_set.append(feature_dict)
            if topic in reuters.categories(file_id):
                test_classifications.append(topic)
            else:
                test_classifications.append("not " + topic)
    vectorizer = DictVectorizer()
    print(training_set[:10])
    training_data = vectorizer.fit_transform(training_set)
    # fixes the total number of features in the model
    test_data = vectorizer.transform(test_set)
    # ignores any features in the test set that weren't in the training set
    return training_data,training_classifications,test_data,test_classifications


trn_data,trn_classes,test_data,test_classes = prepare_reuters_data("acq",get_BOW)


print(trn_classes)


print(trn_data)

print(test_classes)


print(test_data)









# one-class svm for imbalanced binary classification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
print('print X', X)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)

print('print trainX', type(trainX),len(trainX[trainy == 0]),trainX.shape)
print('print testX', len(testX))
print('print trainy', type(trainy),len(trainy),trainy.shape)
print('print testy', len(testy))
# define outlier detection model
model = OneClassSVM(gamma='scale', nu=0.01)
# fit on majority class
a = trainX[trainy == 1]

trainX = trainX[trainy==0]

print('print new trainX', len(trainX))
print('print exception', len(a))
model.fit(trainX)
# detect outliers in the test set
yhat = model.predict(testX)
# mark inliers 1, outliers -1
testy[testy == 1] = -1
testy[testy == 0] = 1
# calculate score
score = f1_score(testy, yhat, pos_label=-1)
print('testy(1):\n',len(testy[testy == 1]))
print('testy(-1):\n',len(testy[testy == -1]))
print('print out yhat',len(yhat[yhat == -1]))
# testy <class 'numpy.ndarray'> 5000 [1 1 1 ... 1 1 1]
# testy(1):4995
# testy(-1):5
# yhat <class 'numpy.ndarray'> 5000 [1 1 1 ... 1 1 1]
# (yhat == -1) 60
print('F1 Score: %.3f' % score)

