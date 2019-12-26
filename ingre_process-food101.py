import io
import scipy.io as matio
import os
import os.path
import numpy as np
from PIL import Image
import time
import re

import opts

opt = opts.opt_ingre_process_food101()


def loadMat(root_path, fileName):
    file = matio.loadmat(root_path + fileName)[fileName[:-4]]
    return file

def loadNpy(root_path, fileName):
    file = np.load(root_path + fileName)
    return file


def parse_ingre_presence(raw_ingre_info):
    # read ingredient data
    with io.open(raw_ingre_info, encoding='utf-8') as file:
        lines = file.read().split('\n')

    # construct ingre dict and the multi-hot ingredient distributions of all 101 classes
    num_class = len(lines)
    max_num_word = 500

    ingreList = []
    ingredient_all_feature = np.zeros((num_class, max_num_word))

    #process each line of ingredients
    for i in range(num_class):
        print('processing line ' + str(i))

        #get a line of ingredients
        line = lines[i]
        ingredients = line.split(',')  # line is a string

        #check each of the ingredient
        num_ingre = len(ingredients)
        for j in range(num_ingre):
            ingredient = ingredients[j]
            if ingredient in ingreList: #fill the feature vector if ingredient in our dict
                ingre_index = ingreList.index(ingredient)
                ingredient_all_feature[i,ingre_index] = 1
            else: # expand dict and fill feature vector
                ingreList.append(ingredient)
                ingredient_all_feature[i, len(ingreList)-1] = 1

    dict_size = len(ingreList)
    ingredient_all_feature = ingredient_all_feature[:,:dict_size]

    matio.savemat(root_path + 'ingreList.mat', {'ingreList': ingreList})
    matio.savemat(root_path + 'ingredient_all_feature.mat', {'ingredient_all_feature': ingredient_all_feature})

    return ingreList, ingredient_all_feature



def get_ingre_term2word_map(root_path, ingreList):
    # initialization
    wordList = []  # record the list of words in ingredients
    ingre2word_map = np.zeros((len(ingreList), 1000))
    num_words = 0  # total counts for individual words

    # create a list for ingredient words
    num_ingre = len(ingreList)
    for i in range(num_ingre):
        print('process ingredient {}'.format(i))
        words = ingreList[i].split()  # get individual words in a gredient

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



def create_LSTM_input(root_path, ingredient_all_feature):
    # Parameters
    max_seq = 30  # The maximum number of words
    num_data = len(ingredient_all_feature)

    # construct indexVectors
    indexVector = np.zeros((num_data, max_seq))  # store the input seq of each class
    seq_max = 0
    seq_avg = 0

    for i in range(0, num_data):  # for ingre vector of each class
        # print('processing data ' + str(i))

        # get the indexes of ingredient terms
        data = ingredient_all_feature[i]
        index_term = np.where(data > 0)[0]

        # fill indexVector
        len_seq = len(index_term)
        indexVector[i, :len_seq] += index_term + 1 #get 1-indexed ingredients
        if len_seq > seq_max:
            seq_max = len_seq
        seq_avg += len_seq

    print('max seq: {}'.format(seq_max))
    print('avg seq: {}'.format(seq_avg / num_data))

    # shorten indexVector to have seq_max in sequence length
    indexVector = indexVector[:, 0:seq_max]

    # save the inputs
    matio.savemat(root_path + 'indexVector.mat', {'indexVector': indexVector})

    return indexVector



def create_glove_matrix(root_path, golve_root_path, wordList):

    #produce glove vectors for our ingredients
    glove_head = loadMat(golve_root_path, 'glove_head.mat')
    glove_vector = loadMat(golve_root_path, 'glove_vector.mat')
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


def get_dataset_statistics(root_path, indexVector, ingredient_all_feature):

    # Parameters
    seq = indexVector.shape[1]  # max length of sequence: 25
    num_cls = len(indexVector)  # number of classes in dataset: 101
    num_ingre = ingredient_all_feature.shape[1]  # number of ingredients in dataset

    # construct output relational matrices
    # Probabilities of 446 words presented in 101 classes
    Pr_cls = np.zeros((num_cls, num_ingre))
    # Cooccurrence robabilities between 446 words in 101 classes
    Pr_co = np.zeros((num_cls, num_ingre, num_ingre))
    # Probabilities of 446 words presented as the s-th word in 101 classes
    Pr_wInS = np.zeros((num_cls, seq, num_ingre))
    # Probabilities of a word being the previous one of the given word in sequence in 101 classes
    Pr_pairP = np.zeros((num_cls, num_ingre, num_ingre))
    # Probabilities of a word being the next one of the given word in sequence in 101 classes
    Pr_pairN = np.zeros((num_cls, num_ingre, num_ingre))

    # compute statistics in terms of classes
    indexVector = indexVector.astype(int)

    for i in range(0, num_cls):
        print('Process class {}'.format(i))

        #process ingre indicator to get Pr_co
        indicator = ingredient_all_feature[i]
            #get cooccurence matrix
        indicator_a = np.expand_dims(indicator, axis=1)
        indicator_b = np.expand_dims(indicator, axis=0)
        indicator_matrix = np.matmul(indicator_a, indicator_b)
            #update statistics
        Pr_co[i] = indicator_matrix


        # process ingre seq to get the other statistics
        seq_cur = 0
        data = indexVector[i]-1
        for id in data:
            # check word presence and update pr_cls
            if id > -1:  # note that id \in [-1,445]
                Pr_cls[i, id] = 1
            else:
                break

            # update Pr_wInS
            Pr_wInS[i, seq_cur, id] = 1

            # update Pr_pairP and Pr_pairN
            if seq_cur > 0:
                Pr_pairP[i, id, data[seq_cur - 1]] = 1
            if seq_cur < seq - 1 and data[seq_cur + 1] > -1:
                Pr_pairN[i, id, data[seq_cur + 1]] = 1

            # count at next word id
            seq_cur += 1

    # get final values
    for i in range(0, num_cls):

        Pr_cls[i, :] = Pr_cls[i, :] / np.sum(Pr_cls[i, :])

        for s in range(0, seq):
            if np.sum(Pr_wInS[i, s, :]) == 0:
                break
            Pr_wInS[i, s, :] = Pr_wInS[i, s, :] / np.sum(Pr_wInS[i, s, :])

        for w in range(0, num_ingre):
            if np.sum(Pr_pairP[i, w, :]) != 0:
                Pr_pairP[i, w, :] = Pr_pairP[i, w, :] / np.sum(Pr_pairP[i, w, :])
            if np.sum(Pr_pairN[i, w, :]) != 0:
                Pr_pairN[i, w, :] = Pr_pairN[i, w, :] / np.sum(Pr_pairN[i, w, :])


    matio.savemat(root_path + 'Pr_cls.mat', {'Pr_cls': Pr_cls})
    matio.savemat(root_path + 'Pr_co.mat', {'Pr_co': Pr_co})
    matio.savemat(root_path + 'Pr_wInS.mat', {'Pr_wInS': Pr_wInS})
    matio.savemat(root_path + 'Pr_pairP.mat', {'Pr_pairP': Pr_pairP})
    matio.savemat(root_path + 'Pr_pairN.mat', {'Pr_pairN': Pr_pairN})

    return Pr_cls, Pr_wInS, Pr_pairP, Pr_pairN




#-----------------------------------------------------------------------------------------------------------------
#parse ingredient presence for samples

#set paths
root_path = opt.data_path
raw_ingre_info = root_path + 'ingredients.txt'

#process ingredient data - create intermediate data
ingreList, ingredient_all_feature = parse_ingre_presence(raw_ingre_info)

#-----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------
#create LSTM input features

indexVector = create_LSTM_input(root_path, ingredient_all_feature)

#-----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------
#get ingredient words

ingre2word_map, wordList = get_ingre_term2word_map(root_path, ingreList)

#-----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------
#match glove vectors for our 461 ingredient words

golve_root_path = opt.golve_root_path

#get embeddings of 461 ingredient words
wordVector_word = loadMat(root_path, 'wordVector_word.mat')
#wordVector_word = create_glove_matrix(root_path, golve_root_path, wordList)

#get embeddings of 446 ingredient terms
wordVector = np.zeros([446,300])

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

Pr_cls, Pr_wInS, Pr_pairP, Pr_pairN = get_dataset_statistics(root_path, indexVector, ingredient_all_feature)

#-----------------------------------------------------------------------------------------------------------------