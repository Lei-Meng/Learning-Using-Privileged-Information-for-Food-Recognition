import io
import scipy.io as matio
import os
import os.path
import numpy as np
from PIL import Image
import time
import re

import opts

opt = opts.opt_ingre_process_vireo()



def loadMat(root_path, fileName):
    file = matio.loadmat(root_path + fileName)[fileName[:-4]]
    return file

def loadNpy(root_path, fileName):
    file = np.load(root_path + fileName)
    return file

aa = loadMat(opt.data_path, 'size_class.mat')

def parse_ingre_presence(raw_ingre_info):
    # read ingredient data
    with io.open(raw_ingre_info, encoding='utf-8') as file:
        lines = file.read().split('\n')[:-1]  # note the last line is empty for '\n'

    # save the image names and their multi-hot ingredient distributions
    ingredient_all_head = []
    ingredient_all_feature = np.zeros((len(lines), 353))

    i = 0
    for line in lines:
        print('processing line ' + str(i))
        a = line.split()[0]  # a is a string
        b = line.split()[1:]
        tmp = np.asarray(b, dtype=int)
        tmp[np.where(tmp < 0)] = 0

        ingredient_all_head.append(a)
        ingredient_all_feature[i, :] = tmp[:]

        i += 1

    matio.savemat(root_path + 'ingredient_all_head.mat', {'ingredient_all_head': ingredient_all_head})
    matio.savemat(root_path + 'ingredient_all_feature.mat', {'ingredient_all_feature': ingredient_all_feature})

    return ingredient_all_head, ingredient_all_feature
    
def build_feature_vectors(root_path,train_data_path,ingredient_all_head, ingredient_all_feature,index_map,mode):
    # process training/val/test data
    with io.open(train_data_path, encoding='utf-8') as file:
        lines = file.read().split('\n')[:-1]

    ingredient_train_feature = np.zeros((len(ingredient_all_head), 353))

    i = 0
    for line in lines:
        print('processing line ' + str(i))
        for j in range(len(index_map)):
            head = ingredient_all_head[index_map[j]]
            if re.match(line, head):
                ingredient_train_feature[i, :] = ingredient_all_feature[index_map[j]]
                index_map = np.delete(index_map, j)
                break
        i += 1

    ingredient_train_feature[np.where(ingredient_train_feature < 0)] = 0
    matio.savemat(root_path + 'ingredient_' + mode +'_feature.mat', {'ingredient_' + mode + '_feature': ingredient_train_feature})
    return ingredient_train_feature


def create_ingre_features(root_path, ingredient_all_head, ingredient_all_feature):
    # set paths
    train_data_path = root_path + 'TR.txt'
    val_data_path = root_path + 'VAL.txt'
    test_data_path = root_path + 'TE.txt'

    # get features
    index_map = np.arange(len(ingredient_all_head))  # limit the search space to the list of all lines of data
    ingredient_train_feature = build_feature_vectors(root_path,train_data_path,ingredient_all_head, ingredient_all_feature,index_map,'train')
    ingredient_val_feature = build_feature_vectors(root_path, val_data_path, ingredient_all_head, ingredient_all_feature, index_map, 'val')
    ingredient_test_feature = build_feature_vectors(root_path, test_data_path, ingredient_all_head, ingredient_all_feature, index_map, 'test')

    return ingredient_train_feature, ingredient_val_feature, ingredient_test_feature


def get_ingre_term2word_map(root_path):
    with open(root_path + 'IngredientList.txt', 'r', encoding="utf8") as f:
        ingre_list = f.read().split('\n')[:-1]

    ingre_list = np.array(ingre_list)
    matio.savemat(root_path + 'ingreList.mat', {'ingreList': ingre_list})

    wordList = []  # record the list of words in ingredients
    ingre2word_map = np.zeros((len(ingre_list), 1000))
    num_words = 0  # total counts for individual words

    for i in range(0, len(ingre_list)):
        print('process word {}'.format(i))
        words = ingre_list[i].split()  # individual words in a gredient

        for word in words:
            if word in wordList:
                ingre2word_map[i, wordList.index(word)] = 1
            else:
                wordList.append(word)
                num_words += 1
                ingre2word_map[i, num_words - 1] = 1
        
    matio.savemat(root_path + 'wordList.mat', {'wordList': wordList})

    ingre2word_map = ingre2word_map[:, 0:num_words]
    matio.savemat(root_path + 'ingre2word_map.mat', {'ingre2word_map': ingre2word_map})
    return ingre2word_map, wordList

def create_ingre_word_features(root_path, ingredient_train_feature, ingredient_val_feature, ingredient_test_feature):#feature_vectors
    ingre2word_map, wordList = get_ingre_term2word_map(root_path)

    # compute the final word indicator for data
    wordIndicator_train = ingredient_train_feature @ ingre2word_map
    wordIndicator_val = ingredient_val_feature @ ingre2word_map
    wordIndicator_test = ingredient_test_feature @ ingre2word_map

    wordIndicator_train[np.where(wordIndicator_train > 1)] = 1
    wordIndicator_val[np.where(wordIndicator_val > 1)] = 1
    wordIndicator_test[np.where(wordIndicator_test > 1)] = 1

    matio.savemat(root_path + 'wordIndicator_train.mat', {'wordIndicator_train': wordIndicator_train})
    matio.savemat(root_path + 'wordIndicator_val.mat', {'wordIndicator_val': wordIndicator_val})
    matio.savemat(root_path + 'wordIndicator_test.mat', {'wordIndicator_test': wordIndicator_test})
    return wordIndicator_train, wordIndicator_val, wordIndicator_test, ingre2word_map, wordList


def get_word_sequence(root_path, ingredient_train_feature, ingre2word_map, mode):

    # Parameters
    max_seq = 30  # The maximum number of words
    num_data = len(ingredient_train_feature)

    # construct indexVectors
    indexVector = np.zeros((num_data, max_seq))  # store the input seq of 353 ingredient terms for each food item
    seq_max = 0
    seq_avg = 0

    indexVector_word = np.zeros((num_data, max_seq))  # store the input seq of 309 ingredient words for each food item
    seq_max_word = 0
    seq_avg_word = 0

    for i in range(0, num_data):  # for each food item
        #print('processing data ' + str(i))

        # get the indexes of ingredient terms
        data = ingredient_train_feature[i, :]
        index_term = np.where(data > 0)[0]

        # fill indexVector
        len_seq = len(index_term)
        indexVector[i,:len_seq] += index_term + 1
        if len_seq > seq_max:
            seq_max = len_seq
        seq_avg += len_seq

        # fill indexVector_word
        seq_cur = 0  # record the length of sequence for current data
            # get the indexes of ingredient words
        for term in index_term:  # for each ingredient term
            indicator_word = ingre2word_map[term, :]
            index_word = np.where(indicator_word > 0)[0]
            for word in index_word:
                indexVector_word[i, seq_cur] = word + 1  # add 1 for two purposes: 1) denote empty entry with 0, 2) facilitate nn.embedding using row 0 for empty entry
                seq_cur += 1

        # update the seq records
        if seq_cur > seq_max_word:
            seq_max_word = seq_cur
        seq_avg_word += seq_cur

    print('max seq: {} {}'.format(seq_max, seq_max_word))
    print('avg seq: {} {}'.format(seq_avg / num_data, seq_avg_word / num_data))

    # shorten indexVector to have seq_max in sequence length
    indexVector= indexVector[:, 0:seq_max]
    indexVector_word = indexVector_word[:, 0:seq_max_word]

    #save the inputs
    matio.savemat(root_path + 'indexVector_' + mode + '.mat', {'indexVector_' + mode: indexVector})
    #matio.savemat(root_path +  'indexVector_word_' + mode + '.mat', {'indexVector_word_' + mode : indexVector_word})

    return indexVector #[indexVector, indexVector_word]


def create_LSTM_input(root_path, ingredient_train_feature, ingredient_val_feature, ingredient_test_feature, ingre2word_map):

    indexVector_train = get_word_sequence(root_path, ingredient_train_feature, ingre2word_map, 'train')
    indexVector_val = get_word_sequence(root_path, ingredient_val_feature, ingre2word_map, 'val')
    indexVector_test = get_word_sequence(root_path, ingredient_test_feature, ingre2word_map, 'test')

    return indexVector_train, indexVector_val, indexVector_test

def create_glove_matrix(root_path, glove_path, wordList):

    #process glove file
    with io.open(glove_path, encoding='utf-8') as file:
        lines = file.read().split('\n')[:-1] #take care of the empty entry!
        if len(lines) != 400000:
            print('error in splitting glove text!!!')

    num_lines = len(lines)
    num_dim = 300

    glove_head = []
    glove_vector = np.zeros((num_lines,num_dim))

    #get words and vectors in glove
    i=0
    for line in lines:
        print("processing {}-th line".format(i))
        line = line.split()
        glove_head.append(line[0])
        glove_vector[i] = line[1:]

        i+=1

    matio.savemat(root_path + 'glove_head.mat', {'glove_head': glove_head})
    matio.savemat(root_path + 'glove_vector.mat', {'glove_vector': glove_vector})

    #produce glove vectors for our ingredients
    num_word = len(wordList)

    p=0 #indicate the index of words in wordList
    wordVector = np.zeros((num_word,300))
    count = 0
    for word in wordList:
        print(p)
        q = 0
        for glove_word in glove_head:
            if re.match(word, glove_word):
                wordVector[p,:] = glove_vector[q,:]
                print('word {} matches glove word {}'.format(p,q))
                count+=1
                break
            q+=1
        p+=1

    print(count)
    matio.savemat(root_path + 'wordVector_word.mat', {'wordVector_word': wordVector})

    return wordVector


def get_dataset_statistics(root_path, indexVector_train, indexVector_val,  ingredients_train, ingredients_val):

    #concate train & val data
        # pad val for the following concatenation. Here we know the sequence len of indexVector_train is larger than indexVector_val
    pad_vec = np.zeros((indexVector_val.shape[0], indexVector_train.shape[1]))
    pad_vec[:, 0:indexVector_val.shape[1]] = indexVector_val[:]
        # concate to obtain the entire data
    indexVector = np.concatenate((indexVector_train, pad_vec), axis=0).astype(np.int64) - 1  # do -1 to facilitate getting non-empty entries

    ingredients = np.concatenate((ingredients_train, ingredients_val), axis=0)


    # load and concate labels
    labels_train = matio.loadmat(root_path + 'train_label.mat')['train_label']
    labels_val = matio.loadmat(root_path + 'val_label.mat')['validation_label']
    labels = np.concatenate((labels_train, labels_val), axis=1)[0, :] - 1  # do -1 to get 0-indexed labels to facilitate following computing

    # Parameters
    seq = indexVector.shape[1]  # max length of sequence
    num_cls = 172  # number of classes in dataset
    num_word = 353  # number of words in dataset
    num_data = len(indexVector)  # number of data in dataset

    # construct output relational matrices
    # Probabilities of 353 words presented in 172 classes
    Pr_cls = np.zeros((num_cls, num_word))
    # Cooccurrence robabilities between 353 words in 172 classes
    Pr_co = np.zeros((num_cls, num_word, num_word))
    # Probabilities of 353 words presented as the s-th word in 172 classes
    Pr_wInS = np.zeros((num_cls, seq, num_word))
    # Probabilities of a word being the previous one of the given word in sequence in 172 classes
    Pr_pairP = np.zeros((num_cls, num_word, num_word))
    # Probabilities of a word being the next one of the given word in sequence in 172 classes
    Pr_pairN = np.zeros((num_cls, num_word, num_word))
    # number of data in 172 classes
    size_class = np.zeros((num_cls))

    # start computing
    for i in range(0, num_data):
        print('Process item {}'.format(i))
        # read data item and its label
        data = indexVector[i]
        label = labels[i]

        # increase class size
        size_class[label] += 1

        #process word indicator to get Pr_co
        indicator = ingredients[i]
            #get cooccurence matrix
        indicator_a = np.expand_dims(indicator, axis=1)
        indicator_b = np.expand_dims(indicator, axis=0)
        indicator_matrix = np.matmul(indicator_a, indicator_b)
            #update statistics
        Pr_co[label] += indicator_matrix


        # process word seq to get other statistics
        seq_cur = 0
        for id in data:
            # check word presence and update pr_cls
            if id > -1:  # note that id \in [-1,352]
                Pr_cls[label, id] += 1
            else:
                break

            # update Pr_wInS
            Pr_wInS[label, seq_cur, id] += 1

            # update Pr_pairP and Pr_pairN
            if seq_cur > 0:
                Pr_pairP[label, id, data[seq_cur - 1]] += 1
            if seq_cur < seq - 1 and data[seq_cur + 1] > -1:
                Pr_pairN[label, id, data[seq_cur + 1]] += 1

            # count at next word id
            seq_cur += 1

    # get final values
    for i in range(0, num_cls):
        Pr_co[i, :] = Pr_co[i, :] / size_class[i]

        Pr_cls[i, :] = Pr_cls[i, :] / np.sum(Pr_cls[i, :])

        for s in range(0, seq):
            if np.sum(Pr_wInS[i, s, :]) == 0:
                break
            Pr_wInS[i, s, :] = Pr_wInS[i, s, :] / np.sum(Pr_wInS[i, s, :])

        for w in range(0, num_word):
            if np.sum(Pr_pairP[i, w, :]) != 0:
                Pr_pairP[i, w, :] = Pr_pairP[i, w, :] / np.sum(Pr_pairP[i, w, :])
            if np.sum(Pr_pairN[i, w, :]) != 0:
                Pr_pairN[i, w, :] = Pr_pairN[i, w, :] / np.sum(Pr_pairN[i, w, :])

    test = 0
    for k in range(172):
        for p in range(353):
            for q in range(p,353):
                test += Pr_co[k,p,q]-Pr_co[k,q,p]


    matio.savemat(root_path + 'size_class.mat', {'size_class': size_class})
    matio.savemat(root_path + 'Pr_cls.mat', {'Pr_cls': Pr_cls})
    matio.savemat(root_path + 'Pr_co.mat', {'Pr_co': Pr_co})
    matio.savemat(root_path + 'Pr_wInS.mat', {'Pr_wInS': Pr_wInS})
    matio.savemat(root_path + 'Pr_pairP.mat', {'Pr_pairP': Pr_pairP})
    matio.savemat(root_path + 'Pr_pairN.mat', {'Pr_pairN': Pr_pairN})

    return size_class, Pr_cls, Pr_wInS, Pr_pairP, Pr_pairN




#-----------------------------------------------------------------------------------------------------------------
#parse ingredient presence for samples

#set paths
root_path = opt.data_path
raw_ingre_info = root_path + 'IngreLabel.txt'

#process ingredient data - create intermediate data
ingredient_all_head, ingredient_all_feature = parse_ingre_presence(raw_ingre_info)

#-----------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------
#create features (353 ingredient terms) of train/val/test data

ingredient_train_feature, ingredient_val_feature, ingredient_test_feature = create_ingre_features(root_path, ingredient_all_head, ingredient_all_feature)

#-----------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------
#create features (309 ingredient words) of train/val/test data

wordIndicator_train, wordIndicator_val, wordIndicator_test, ingre2word_map, wordList = create_ingre_word_features(root_path, ingredient_train_feature, ingredient_val_feature, ingredient_test_feature)

#-----------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------
#create LSTM input features (353/309 ingredient terms/words) of train/val/test data

indexVector_train, indexVector_val, indexVector_test = create_LSTM_input(root_path, ingredient_train_feature, ingredient_val_feature, ingredient_test_feature, ingre2word_map)

#-----------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------
#match glove vectors for our 309 ingredient words

glove_path = root_path + 'glove.6B.300d.txt'

#get embeddings of 309 ingredient words
wordVector_word = create_glove_matrix(root_path, glove_path, wordList)

#get embeddings of 353 ingredient terms
wordVector = np.zeros([353,300])

#process each ingredient term
for i in range(wordVector.shape[0]):
    #get the ingre words for the i-th ingredient term
    ingre_word_indicator = ingre2word_map[i]
    index_words = np.where(ingre_word_indicator>0)[0]

    #get ingre term embedding by sum of those of words
    wordVector[i] += wordVector_word[index_words].sum(0)

matio.savemat(root_path + 'wordVector.mat', {'wordVector': wordVector})
#-----------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------
#get dataset statistics on class-ingredient and ingredient-ingredient relations as prior knowledge

size_class, Pr_cls, Pr_wInS, Pr_pairP, Pr_pairN = get_dataset_statistics(root_path, indexVector_train, indexVector_val, ingredient_train_feature, ingredient_val_feature)

#-----------------------------------------------------------------------------------------------------------------