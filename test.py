import io
import scipy.io as matio
import os
import os.path
import numpy as np
from PIL import Image
import time
import re

import torch
import torch.utils.data
import torch.nn.parallel as para
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.utils.model_zoo as model_zoo



#load environmental settings
import opts
opt = opts.opt_algorithm()



# —Path settings———————————————————————————————————————————————————————————————————————————————————————————————————————

#file paths
def prepare_intermediate_folders(path):
    if not os.path.exists(path):
        os.makedirs(path)

dataset_indicator = opt.dataset
root_path = opt.root_path
image_path = opt.img_path
data_path = opt.data_path
result_path = opt.result_path


#train type
train_stage = opt.stage
mode = opt.mode
img_net_type = opt.img_net
ingre_net_type = opt.ingre_net


#data info
num_class = opt.dataset_num_class
num_word = opt.dataset_num_ingre
avg_word = opt.dataset_avg_ingre
if dataset_indicator is 'vireo':
    if mode is 'test':
        max_seq = opt.dataset_max_seq_test
    else:
        max_seq = opt.dataset_max_seq_val
else:
    max_seq = opt.dataset_max_seq

#Used in performance estimation for ingredient prediction
if dataset_indicator is 'vireo':
    with io.open(opt.food_class_name_path, encoding='utf-8') as file:
        class_names = file.read().split('\n')[:-1]
    class_names = np.array(class_names)

    ingre_names = matio.loadmat(opt.ingre_term_list_path)['ingreList']
    for i in range(len(ingre_names)):
        ingre_names[i] = ingre_names[i].strip()
else:
    with io.open(opt.food_class_name_path, encoding='utf-8') as file:
        class_names = file.read().split('\n')
    class_names = np.array(class_names)

    ingre_names = matio.loadmat(opt.ingre_term_list_path)['ingreList']
    for i in range(len(ingre_names)):
        ingre_names[i] = ingre_names[i].strip()

# Image Info
no_of_channels = 3
image_size = [256, 256]  # [64,64]


#algorithm hyperparameter
CUDA = 1  # 1 for True; 0 for False
SEED = 1
BATCH_SIZE = 2
LOG_INTERVAL = 1

if img_net_type is 'vgg19bn':
    latent_len = 4096
else:
    latent_len = 2048
blk_len = int(latent_len * 3 / 8)

lr_decay = 4
EPOCHS = lr_decay * 3 + 1

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)
kwargs = {'num_workers': 2, 'pin_memory': True} if CUDA else {}

#--Create dataset and dataloader----------------------------------------------------------------------------------

#define transform ops
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_img = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

#create dataset
from build_dataset import build_dataset
dataset = build_dataset(train_stage, image_path, data_path, transform_img, mode, dataset_indicator)

#dataloader
test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)


#--Model----------------------------------------------------------------------------------

from mymodel import build_mymodel
model, _ = build_mymodel(mode, data_path, CUDA, image_size, latent_len, blk_len, num_class, max_seq, num_word,\
                                 None, None, train_stage, [img_net_type, ingre_net_type], None, None)
model.eval()

# --performance----------------------------------------------------------------

from compute_performance import compute_performance



# --test & validation----------------------------------------------------------------

def print_state(state):
    print(state + '/n')
    # records current progress for tracking purpose
    with io.open(result_path + 'model_batch_test_performance.txt', 'a', encoding='utf-8') as file:
        file.write(state + '/n')

def compute_distribution_similarity(predicted_distribution, true_distribution):
    if len(predicted_distribution.shape) > 1:
        predicted_distribution = predicted_distribution.flatten()
        true_distribution = true_distribution.flatten()

    concated_distributions = np.stack([predicted_distribution, true_distribution],0)
    min = np.min(concated_distributions, 0)
    max = np.max(concated_distributions, 0)
    return np.sum(min) / np.sum(max) if np.sum(max) else 0

# def predicts_normalization(predicts):
#     #perform min-max normalization and then sum-to-one
#     for i in range(len(predicts)):# for each batch item
#         min = predicts[i].min()
#         max = predicts[i].max()
#         predicts[i] = (predicts[i] - min) / (max - min)
#         predicts[i] = predicts[i] / predicts[i].sum()
#     return predicts

def compute_ingre_presence(x2y_ingre_recon, Top_n_for_predicts, top_n_ingredients, num_ingredients):
    # create matrix for intermediate results
    matrix_for_presence = np.zeros([top_n_ingredients, num_ingredients])

    for j in range(top_n_ingredients):  # for each of top-n seq entry
        # get the top-3 predicts and convert to ingre prediction vector
        sorted_predicts = (-x2y_ingre_recon[j, :]).argsort()  # ranked list of indexes, value:low -> high
        Top_predicts_index = sorted_predicts[:Top_n_for_predicts]
        Top_predicts_prob = x2y_ingre_recon[j, Top_predicts_index]

        matrix_for_presence[j, Top_predicts_index] = Top_predicts_prob
    return np.max(matrix_for_presence, 0)

def compute_cls_predicts_presence(top_cls_index, Pr_cls, top_n_class, Pr_ingre_presence):
    # get top-10 predicted classes
    top_cls_ingre_statistics = Pr_cls[top_cls_index]

    # compute probabilities of matching between Pr_ingre_presence and top-10 cls statistics
    ingre_infer_predicts_presence = np.zeros(top_n_class)
    for k in range(top_n_class):
        ingre_infer_predicts_presence[k] = compute_distribution_similarity(Pr_ingre_presence, top_cls_ingre_statistics[k])
    return ingre_infer_predicts_presence


def compute_ingre_coocurrence(Pr_ingre_presence):
    #compute coocurrence
    indicator_a = np.expand_dims(Pr_ingre_presence, axis=1)
    indicator_b = np.expand_dims(Pr_ingre_presence, axis=0)
    indicator_matrix = np.matmul(indicator_a, indicator_b)
    return indicator_matrix

def compute_cls_predicts_coocurrence(top_cls_index, Pr_co, top_n_class, Pr_ingre_coocurrence):
    # get top-10 predicted classes
    top_cls_ingre_statistics = Pr_co[top_cls_index]

    # compute probabilities of matching between Pr_ingre_presence and top-10 cls statistics
    ingre_infer_predicts_coocurrence = np.zeros(top_n_class)
    for k in range(top_n_class):
        ingre_infer_predicts_coocurrence[k] = compute_distribution_similarity(Pr_ingre_coocurrence, top_cls_ingre_statistics[k])

    return ingre_infer_predicts_coocurrence


def compute_ingre_sequence(x2y_ingre_recon, Top_n_for_predicts, top_n_ingredients, num_ingredients):
    # create matrix for intermediate results
    matrix_for_ingre_order = np.zeros([top_n_ingredients, num_ingredients, num_ingredients])

    for j in range(top_n_ingredients-1):  # for each of top-(n-1) seq entry
        # get the top-3 predicts from current hidden vector
        sorted_predicts = (-x2y_ingre_recon[j, :]).argsort()  # ranked list of indexes, value:low -> high
        Top_predicts_index_cur = sorted_predicts[:Top_n_for_predicts]
        Top_predicts_prob_cur = x2y_ingre_recon[j, Top_predicts_index_cur]

        # get the top-3 predicts from next hidden vector
        sorted_predicts = (-x2y_ingre_recon[j+1, :]).argsort()
        Top_predicts_index_next = sorted_predicts[:Top_n_for_predicts]
        Top_predicts_prob_next = x2y_ingre_recon[j, Top_predicts_index_next]

        #fill the matrix for ingredient orders
        for p in range(Top_n_for_predicts):
            for q in range(Top_n_for_predicts):
                matrix_for_ingre_order[j,Top_predicts_index_cur[p],Top_predicts_index_next[q]] = Top_predicts_prob_cur[p] * Top_predicts_prob_next[q]

    #get class wise ingre order
    class_wise_ingre_order = np.max(matrix_for_ingre_order, 0)
    for k in range(num_ingredients):
        denominator = class_wise_ingre_order[k].sum()
        if denominator > 0:
            class_wise_ingre_order[k] = class_wise_ingre_order[k] / denominator

    return class_wise_ingre_order

def compute_cls_predicts_sequence(top_cls_index, Pr_pairN, top_n_class, Pr_ingre_sequence):
    # get top-10 predicted classes
    top_cls_ingre_statistics = Pr_pairN[top_cls_index]

    # compute probabilities of matching between Pr_ingre_presence and top-10 cls statistics
    ingre_infer_predicts_sequence = np.zeros(top_n_class)
    for k in range(top_n_class):
        ingre_infer_predicts_sequence[k] = compute_distribution_similarity(Pr_ingre_sequence, top_cls_ingre_statistics[k])
    return ingre_infer_predicts_sequence

def get_softmax(numpy_vector):
    tensor = torch.from_numpy(numpy_vector).unsqueeze(0)
    tensor = F.softmax(tensor, dim=1)
    return tensor.numpy()[0]

def get_fused_predicts(predicts_t, predicts_v):
    predicts_t = get_softmax(predicts_t)
    predicts_v = get_softmax(predicts_v)

    return np.min(np.stack([predicts_t,predicts_v],0), 0)


def get_top_n_predicts_nn(x2y_ingre_recon, top_n_ingredients):
    sorted_predicts = (-x2y_ingre_recon).argsort()  # ranked list of indexes, value:low -> high
    Top_predicts_index = sorted_predicts[:top_n_ingredients]
    Top_predicts_prob = x2y_ingre_recon[Top_predicts_index]

    filtered_ingre_predicts = np.zeros(len(x2y_ingre_recon))
    filtered_ingre_predicts[Top_predicts_index] = Top_predicts_prob
    return filtered_ingre_predicts




def get_fused_decision(predicts_v, x2y_ingre_recon, avg_word, ingre_net_type, data_path):
    #user-defined n of top-n predicts for matching
    top_n_class = 10 # match top-10 classes in predicts_v
    top_n_ingredients = avg_word # consider first avg_word words in seq

    #batch info
    batch_size = len(predicts_v)

    #load dataset class-ingredient and ingredient-ingredient statistics
    Pr_cls = matio.loadmat(data_path + 'Pr_cls.mat')['Pr_cls']
    Pr_co = matio.loadmat(data_path + 'Pr_co.mat')['Pr_co']
    Pr_pairN = matio.loadmat(data_path + 'Pr_pairN.mat')['Pr_pairN']

    #get the info of top-n predicted classes
    predicts_v = predicts_v.detach().cpu().numpy()
    top_cls_indexes = (-predicts_v).argsort()[:, :top_n_class] #order: high -> low
    top_cls_probs = np.zeros([batch_size, top_n_class])
    for i in range(batch_size):
        top_cls_probs[i] = predicts_v[i][top_cls_indexes[i]]

    #initialize final results
    fused_predicts = np.zeros([batch_size, top_n_class])


    #get ingre-inferred decisions
    if ingre_net_type is 'gru': #x2y_ingre_recon: (seq, batch, number_of_ingredients)
        # consider top-3 of the predictions for each seq entry
        Top_n_for_predicts = 3

        #manual settings for decision fusion
        alpha = 0.5
        beta = 0.5
        gammar = 0.2

        # make x2y_ingre_recon of (batch, seq, number_of_ingredients)
        x2y_ingre_recon = x2y_ingre_recon.transpose(0, 1)
        x2y_ingre_recon = x2y_ingre_recon.cpu().data.numpy()

        num_ingredients = x2y_ingre_recon.shape[2]

        #process each batch item
        for i in range(batch_size):
            # compute Pr_cls-inferred class distribution
                # compute ingre presence from ingre predicts
            Pr_ingre_presence = compute_ingre_presence(x2y_ingre_recon[i], Top_n_for_predicts, top_n_ingredients, num_ingredients)
                # compute Pr_cls-inferred top-10 class predicts
            ingre_infer_predicts_presence = compute_cls_predicts_presence(top_cls_indexes[i], Pr_cls, top_n_class, Pr_ingre_presence)


            #compute Pr_co-inferred class distribution
                # compute ingre coocurrence from ingre predicts
            Pr_ingre_coocurrence = compute_ingre_coocurrence(Pr_ingre_presence)
                # compute Pr_co-inferred top-10 class predicts
            ingre_infer_predicts_coocurrence = compute_cls_predicts_coocurrence(top_cls_indexes[i], Pr_co, top_n_class, Pr_ingre_coocurrence)

            # compute Pr_pairN-inferred class distribution
                # compute ingre sequential dependency from ingre predicts
            Pr_ingre_sequence = compute_ingre_sequence(x2y_ingre_recon[i], Top_n_for_predicts, top_n_ingredients, num_ingredients)
                # compute Pr_co-inferred top-10 class predicts
            ingre_infer_predicts_sequence = compute_cls_predicts_sequence(top_cls_indexes[i], Pr_pairN, top_n_class, Pr_ingre_sequence)

            #get fused decision
            ingre_infer_predicts_fused =alpha * get_softmax(ingre_infer_predicts_presence) + beta * get_softmax(ingre_infer_predicts_coocurrence) + gammar * get_softmax(ingre_infer_predicts_sequence)
            ingre_infer_predicts_fused = ingre_infer_predicts_fused / (alpha + beta + gammar)

            #get final fused predicts
            fused_predicts[i] = get_fused_predicts(ingre_infer_predicts_fused, top_cls_probs[i])

        return fused_predicts, top_cls_indexes

    else: # ingre_net_type is 'nn'
        #manual settings for decision fusion
        alpha = 0.4
        beta = 1-alpha

        x2y_ingre_recon = x2y_ingre_recon.cpu().data.numpy()

        #process each batch item
        for i in range(batch_size):
            #filter out top-n words for later matching
            filtered_ingre_predicts = get_top_n_predicts_nn(x2y_ingre_recon[i], top_n_ingredients)

            # compute Pr_cls-inferred class distribution
            ingre_infer_predicts_presence = compute_cls_predicts_presence(top_cls_indexes[i], Pr_cls, top_n_class, filtered_ingre_predicts)

            #compute Pr_co-inferred class distribution
                # compute ingre coocurrence from ingre predicts
            Pr_ingre_coocurrence = compute_ingre_coocurrence(filtered_ingre_predicts)
                # compute Pr_co-inferred top-10 class predicts
            ingre_infer_predicts_coocurrence = compute_cls_predicts_coocurrence(top_cls_indexes[i], Pr_co, top_n_class, Pr_ingre_coocurrence)

            #get fused decision
            ingre_infer_predicts_fused =alpha * get_softmax(ingre_infer_predicts_presence) + beta * get_softmax(ingre_infer_predicts_coocurrence)

            #get final fused predicts
            fused_predicts[i] = get_fused_predicts(ingre_infer_predicts_fused, top_cls_probs[i])

        return fused_predicts, top_cls_indexes

def compute_cls_precision_for_fused_predicts(predicts, cls_indexes, labels):
    sorted_predicts = (-predicts).argsort()
    top1_labels = sorted_predicts[:, 0].copy()
    batch_size = len(labels)
    for i in range(batch_size):
        top1_labels[i] = cls_indexes[i,top1_labels[i]]
    match = float(sum(top1_labels - labels + 1 == 0))

    top5_labels = sorted_predicts[:, :5]
    hit = 0
    for i in range(batch_size):
        hit += (labels[i] - 1) in cls_indexes[i, top5_labels[i]]

    return match, hit



def test_epoch(epoch):

    print('Testing starts..')

    #for model performance
    top1_accuracy_total_V = 0
    top5_accuracy_total_V = 0
    top1_accuracy_total_T = 0
    top5_accuracy_total_T = 0

    top1_accuracy_total_fuse = 0
    top5_accuracy_total_fuse = 0

    if ingre_net_type is 'gru':
        ingre_pred_precision_total = np.zeros(max_seq)
        ingre_pred_recall_total = np.zeros(max_seq)
        ingre_pred_word_total = np.zeros(5)
    else:
        ingre_pred_precision_total = np.zeros(avg_word)
        ingre_pred_recall_total = np.zeros(avg_word)

    total_time = time.time()

    for batch_idx, (data) in enumerate(test_loader):

        start_time = time.time()
        if batch_idx == 2:
            break

        # load data
        [imgs, indexVectors, ingredients, labels] = data
        if CUDA:
            imgs = imgs.cuda()
            indexVectors = indexVectors.cuda()
            ingredients = ingredients.cuda()


        # perform model inference
        if ingre_net_type is 'gru':
            predicts, _, _, cross_vectors = model(imgs, indexVectors)
        else:
            predicts, _, _, cross_vectors = model(imgs, ingredients)


        # perform decision fusion
        predicts_v = predicts[0]
        x2y_ingre_recon = cross_vectors[1]
        fused_predicts, top_cls_indexes = get_fused_decision(predicts_v, x2y_ingre_recon, avg_word, ingre_net_type, data_path)

        # compute performance
        [top1_hit_V, top5_hit_V], [top1_hit_T, top5_hit_T], ingre_pred_performance = compute_performance(mode, predicts, x2y_ingre_recon, labels, indexVectors, ingredients, ingre_net_type, avg_word, class_names, ingre_names, result_path)

        top1_hit_fuse, top5_hit_fuse = compute_cls_precision_for_fused_predicts(fused_predicts, top_cls_indexes, labels)

        # img cls precision
        top1_accuracy_total_V += top1_hit_V
        top1_accuracy_cur_V = top1_hit_V / float(len(labels))

        top5_accuracy_total_V += top5_hit_V
        top5_accuracy_cur_V = top5_hit_V / float(len(labels))

        # ingre cls precision
        top1_accuracy_total_T += top1_hit_T
        top1_accuracy_cur_T = top1_hit_T / float(len(labels))

        top5_accuracy_total_T += top5_hit_T
        top5_accuracy_cur_T = top5_hit_T / float(len(labels))

        # fused cls precision
        top1_accuracy_total_fuse += top1_hit_fuse
        top1_accuracy_cur_fuse = top1_hit_fuse / float(len(labels))

        top5_accuracy_total_fuse += top5_hit_fuse
        top5_accuracy_cur_fuse = top5_hit_fuse / float(len(labels))

        #ingredient prediction performance
        if ingre_net_type is 'gru':
            [avg_precision, avg_recall, avg_precision_word] = ingre_pred_performance
            ingre_pred_precision_total += avg_precision
            ingre_pred_recall_total += avg_recall
            ingre_pred_word_total += avg_precision_word
        else: #if ingre_net_type is 'nn'
            [avg_precision, avg_recall] = ingre_pred_performance
            ingre_pred_precision_total += avg_precision
            ingre_pred_recall_total += avg_recall


        #print batch-level performance
        if batch_idx % LOG_INTERVAL == 0:
            state = 'Test Epoch: {} [{}/{} ({:.0f}%)] | Top1_v: {:.4f} | Top5_v: {:.4f} | Top1_f: {:.4f} | Top5_f: {:.4f} | Top1_t: {:.4f} | Top5_t: {:.4f} | Time:{} | Total_Time:{}'.format(
                epoch, (batch_idx + 1) * len(ingredients), len(test_loader.dataset), 100. * (batch_idx + 1) / len(test_loader),
                top1_accuracy_cur_V, top5_accuracy_cur_V,
                top1_accuracy_cur_fuse, top5_accuracy_cur_fuse,
                avg_precision[0],
                top1_accuracy_cur_T, top5_accuracy_cur_T,
                round((time.time() - start_time), 4) * LOG_INTERVAL,
                round((time.time() - total_time), 4))
            print_state(state)

    #record epoch-level results
    print('====> Epoch: {} | Top1_Acc_V: {:.4f} | Top5_Acc_V: {:.4f} | Top1_Acc_F: {:.4f} | Top5_Acc_F: {:.4f} | Ingre_Pred_P: {:.4f} | Top1_Acc_T: {:.4f} | Top5_Acc_T: {:.4f} | Total_Time:{}\n'.format(
        epoch,
        top1_accuracy_total_V / len(test_loader.dataset), top5_accuracy_total_V / len(test_loader.dataset),
        top1_accuracy_total_fuse / len(test_loader.dataset), top1_accuracy_total_fuse / len(test_loader.dataset),
        ingre_pred_precision_total[0] / len(test_loader),
        top1_accuracy_total_T / len(test_loader.dataset), top5_accuracy_total_T / len(test_loader.dataset),
        round((time.time() - total_time), 4)))

    with io.open(result_path + 'test_performance.txt', 'a', encoding='utf-8') as file:
        file.write('Epoach {}:\n Top1_Acc_V={:.4f}, Top5_Acc_V={:.4f},\n Top1_Acc_F={:.4f}, Top5_Acc_F={:.4f},\n ingre_p={},\n ingre_r={}'.format(
            epoch,
            top1_accuracy_total_V / len(test_loader.dataset), top5_accuracy_total_V / len(test_loader.dataset),
            top1_accuracy_total_fuse / len(test_loader.dataset), top1_accuracy_total_fuse / len(test_loader.dataset),
            ingre_pred_precision_total / len(test_loader),
            ingre_pred_recall_total / len(test_loader),
        ))

        if ingre_net_type is 'gru':
            file.write(',\n ingre_w = {}\n\n'.format(ingre_pred_word_total / len(test_loader)))
        else:
            file.write('\n\n')

    return top1_accuracy_total_fuse / len(test_loader.dataset), top1_accuracy_total_fuse / len(test_loader.dataset)


# excute main
max_index = 0
max_top1 = 0
max_top5 = 0

start_epoch = lr_decay + 1

for i in range(start_epoch, EPOCHS+1):

    path = result_path + 'model' + str(i) + '.pt'
    trained_model = torch.load(path, map_location='cpu')
    model.load_state_dict(trained_model)

    top1, top5 = test_epoch(i)

    if top1 > max_top1:
        max_top1 = top1
        max_top5 = top5
        max_index = i
print('Max is achieved by model{} with Top1:{} | Top5:{} |'.format(max_index, max_top1, max_top5))



