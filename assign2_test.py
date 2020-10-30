import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet

nltk.download("brown")
nltk.download("wordnet")

# filtered_gold_standard stores the word pairs and their human-annotated similarity in your filtered test set
filtered_gold_standard = {}
set1 = open("set1.tab")
paras = brown.paras()

# lemmatizer
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma



brown_corpus = []
for para in paras:
    new_para = set()
    for line in para:
        for word in line:
            if word.isalpha():
                modified_word = word.lower()
                modified_word = lemmatize(modified_word)
                new_para.add(modified_word)
    brown_corpus.append(new_para)


for line in set1:
    # type(line) = str
    splited_line = line.split( )
    if splited_line[0] == "Word":
        continue
    length0 = len([True for line in brown_corpus if splited_line[0] in line])
    length1 = len([True for line in brown_corpus if splited_line[1] in line])
    if length1 >= 8 and length0 >= 8:
        value = float(splited_line[2])
        filtered_gold_standard[splited_line[0],splited_line[1]] = value

print(len(filtered_gold_standard))
print(filtered_gold_standard)


'''
======================================================== Q2 ========================================================
you should remove any words which do not have a single primary sense
  single primary sense:either  (a) having only one sense (i.e. only one synset)
                        or     (b) where the count (as provided by the WordNet count() method for the lemmas associated 
                               with a synset) of the most common sense is at least 4 times larger than the next most 
                               common sense.  
 
'''
# final_gold_standard stores the word pairs and their human-annotated similarity in your final filtered test set
final_gold_standard = {}
word_primarysense = {} #a dictionary of (word, primary_sense) (used for next section); primary_sense is a synset


def is_SinglePrimarySense (word):
    syns = wordnet.synsets(word)
    if len(syns) == 1 and syns[0].pos() == 'n':
        word_primarysense[word] = syns[0]
        return True
    max_count = []
    for synset in syns:
        lemmas = synset.lemmas()
        count = 0
        for lemma in lemmas:
            if lemma.name() == word:
                count += lemma.count()
        max_count.append(count)
    largest = max(max_count)
    index = max_count.index(largest)
    max_count.sort()
    second_max = max_count[-2]

    if largest == 0 and second_max == 0:
        return False

    if largest >= 4 * second_max and syns[index].pos() == 'n':
        word_primarysense[word] = syns[index]
        return True
    return False

for word in filtered_gold_standard:
    (key1,key2) = word
    if is_SinglePrimarySense(key1) and is_SinglePrimarySense(key2):
        final_gold_standard[key1,key2] = filtered_gold_standard[key1,key2]

print(len(final_gold_standard))
print(final_gold_standard)
print(word_primarysense)

'''
======================================================== Q3 ========================================================  

'''
from nltk.corpus import wordnet_ic
nltk.download('wordnet_ic')
brown_ic = wordnet_ic.ic('ic-brown.dat')
# lin_similarities stores the word pair and Lin similarity mappings
lin_similarities = {}

for word in final_gold_standard:
    (key1,key2) = word
    syns1 = word_primarysense[key1]
    syns2 = word_primarysense[key2]
    score = syns1.lin_similarity(syns2,brown_ic)
    lin_similarities[word] = score
print(lin_similarities)

'''
======================================================== Q4 ========================================================  

'''
import math
# NPMI_similarities stores the word pair and NPMI similarity mappings
NPMI_similarities = {}

def npmi_calculator (word):
    number_paras = 0
    para_number = 0
    para_index = []
    for para in brown_corpus:
        para_number += 1
        if word in para:
            number_paras += 1
            para_index.append(para_number)

    return number_paras, para_index

for pair in final_gold_standard:
    (key1, key2) = pair
    (number1, list1) = npmi_calculator(key1)
    (number2, list2) = npmi_calculator(key2)
    common_num = len(list(set(list1).intersection(list2)))
    if common_num == 0:
        value = -1
    elif math.log2(common_num) == 0:
        value = -1
    else:
        value = (math.log2(number1*number2)/math.log2(common_num))-1
    NPMI_similarities[key1, key2] = value

print(NPMI_similarities)

'''
======================================================== Q5 ========================================================  

'''
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
# LSA_similarities stores the word pair and LSA similarity mappings
LSA_similarities = {}
vec = DictVectorizer()
svd = TruncatedSVD(n_components=500)

texts = []
for para in brown_corpus:
    a = {}
    for word in para:
        a[word] = 1
    texts.append(a)

brown_matrix = vec.fit_transform(texts)
brown_matrix_transposed = brown_matrix.transpose()
brown_matrix_transposed = svd.fit_transform(brown_matrix_transposed)

def cos_sim(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))

for pair in final_gold_standard:
    (key1,key2) = pair
    v1 = brown_matrix_transposed[vec.vocabulary_[key1]]
    v2 = brown_matrix_transposed[vec.vocabulary_[key2]]
    score = cos_sim(v1,v2)
    LSA_similarities[key1,key2] = score
print(LSA_similarities)

'''
======================================================== Q6 ========================================================  
keep lower and frequency >= 5
'''
from collections import Counter
num_train = 12000
UNK_symbol = "<UNK>"
vocab = set([UNK_symbol])
wordlist = []
i = 0
for para in paras:
    i += 1
    if i == num_train:
        break
    for lines in para:
        for word in lines:
            word = word.lower()
            wordlist.append(word)
word_freq = Counter(wordlist)
threshold = 5
filtered = {x : word_freq[x] for x in word_freq if word_freq[x] >= threshold}

for x,v in filtered.items():
    vocab.add(x)

print(len(vocab))

'''
======================================================== Q7 ========================================================  
'''
to_id = dict.fromkeys(vocab)
i = 0
for id in to_id:
    to_id[id] = i
    i += 1

def word_to_id (word):
    if word in to_id.keys():
        id = to_id[word]
    else:
        id = to_id['<UNK>']
    return id

train_input = []
train_output = []
dev_input = []
dev_output = []
iter = 0
for para in paras:
    # lines_input = []
    # lines_output = []
    l = 0
    for line in para:
        if len(line) >= 3:
            for indx in range(0, len(line) - 2):
                pair_input = []
                input1_id = word_to_id(paras[iter][l][indx])
                input2_id = word_to_id(paras[iter][l][indx + 1])
                pair_input.append(input1_id)
                pair_input.append(input2_id)
                output_id = word_to_id(paras[iter][l][indx + 2])
                if iter <= 11999:
                    train_input.append(pair_input)
                    train_output.append(output_id)
                else:
                    dev_input.append(pair_input)
                    dev_output.append(output_id)
        l +=1

    iter += 1
x_train = np.array(train_input)
y_train = np.array(train_output)
x_dev = np.array(dev_input)
y_dev = np.array(dev_output)

print(x_train.shape)
print(y_train.shape)
print(x_dev.shape)
print(y_dev.shape)

'''
======================================================== Q8 ========================================================  
 word embeddings and  ℎ  to 100
 train your model with 3 epochs with a batch size of 256
'''
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from numpy.linalg import norm

lm_similarities = {}
embedding_dim = 100
maxlen = 30
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_dev = pad_sequences(x_dev, padding='post', maxlen=maxlen)



#model definition
model = Sequential(name="feedforward-sequence-input")
model.add(layers.Embedding(input_dim= 50000,output_dim=embedding_dim,input_shape=(30,)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(20000, activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, verbose=True, validation_data=(x_dev, y_dev), batch_size=256)
loss, accuracy = model.evaluate(x_dev, y_dev, verbose=False)
embeddings = model.get_layer(index=0).get_weights()[0]

for pair in final_gold_standard:
    (key1,key2) = pair
    v1 = embeddings[word_to_id(key1)]
    v2 = embeddings[word_to_id[key2]]
    score = cos_sim(v1,v2)
    lm_similarities[key1,key2] = score

print(lm_similarities)

'''
======================================================== Q8 ========================================================  
 word embeddings and  ℎ  to 100
 train your model with 3 epochs with a batch size of 256
'''
from scipy.stats import pearsonr

# pearson_correlations stores the pearson correlations with the gold standard of 'lin', 'NPMI', 'LSA', 'lm'
pearson_correlations = {}
def convert_to_list (gold_dict,dict1,dict2,dict3,dict4):
    gold_dict_list=[]
    dict1_list=[]
    dict2_list=[]
    dict3_list=[]
    dict4_list=[]
    for pair in brown_corpus:
        gold_dict_list.append(gold_dict[pair])
        dict1_list.append(dict1[pair])
        dict2_list.append(dict2[pair])
        dict3_list.append(dict3[pair])
        dict4_list.append(dict4[pair])
    return gold_dict_list,dict1_list,dict2_list,dict3_list,dict4_list
goal,lin,npmi,lsa,lm=convert_to_list(final_gold_standard,lin_similarities,NPMI_similarities,LSA_similarities,lm_similarities)
lin_score = pearsonr(lin,goal)
NPMI_score = pearsonr(npmi,goal)
LSA_score = pearsonr(lsa,goal)
lm_score = pearsonr(lm,goal)
pearson_correlations['lin'] = lin_score
pearson_correlations['NPMI'] = NPMI_score
pearson_correlations['LSA'] = LSA_score
pearson_correlations['lm'] = lm_score
print(pearson_correlations)