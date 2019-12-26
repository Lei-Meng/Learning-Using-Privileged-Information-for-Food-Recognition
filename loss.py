import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F



criterion_cls = nn.CrossEntropyLoss()
criterion_kl = nn.KLDivLoss()
criterion_recon = nn.MSELoss()
criterion_lstm = nn.BCELoss()
softmax = nn.Softmax(dim=1)


def getLSTMloss(gru_predicts, indexVectors, encoder_t_embeds, decoder_t_embeds, CUDA):
    lstm_loss = 0
    embed_match_loss = 0
    gru_predicts = gru_predicts.transpose(0, 1)  # (seq, batch, num_word) -> (batch, seq, num_word)
    encoder_t_embeds = encoder_t_embeds.transpose(0, 1)
    decoder_t_embeds = decoder_t_embeds.transpose(0, 1)
    indexVectors = indexVectors.cpu().numpy() # (batch, seq)

    total_num = 0 #number of seq entries in the batch

    for i in range(0, gru_predicts.shape[0]):  # for each batch item
        item_seq = indexVectors[i, :]
        seq_len = len(np.where(item_seq > 0)[0])  # compute the index of non-zero entries
        
        for j in range(0, seq_len):  # for each seq item
            total_num += 1

            GT = torch.zeros([1, gru_predicts.shape[2]])
            GT[0, item_seq[j]-1] = 1 #note: ingre in seq is 1-indexed, while that in GT is 0-indexed
            if CUDA:
                GT = GT.cuda()

            lstm_loss += criterion_lstm(softmax(gru_predicts[i, j, :].unsqueeze(0)), GT)

            embed_match_loss += torch.sum((encoder_t_embeds[i, j, :] - decoder_t_embeds[i, j, :]) ** 2) / encoder_t_embeds.numel()

    return lstm_loss / total_num, embed_match_loss / total_num



def getATTloss(multi_attention, CUDA):
    Identity = torch.eye(multi_attention.shape[1]).unsqueeze(0)  # (1,seq,seq)
    Identity = Identity.repeat(multi_attention.shape[0], 1, 1)  # (batch,seq,seq)
    ones = torch.ones(multi_attention.shape[1], multi_attention.shape[1])
    ones = ones.repeat(multi_attention.shape[0], 1, 1)  # (batch,seq,seq)

    if CUDA:
        Identity = Identity.cuda()
        ones = ones.cuda()

    multi_attention_Transpose = torch.transpose(multi_attention, 1, 2).contiguous()  # (batch, num_key_ingredient, seq)

    ATT = torch.sum((multi_attention.bmm(multi_attention_Transpose) * (ones - Identity)) ** 2) / (multi_attention.shape[0]*multi_attention.shape[1]*multi_attention.shape[1])

    return ATT

def loss_function_stage1(model_output, groundtruth):
    # image channel loss
    x_recon = model_output
    data = groundtruth
    RE_V = criterion_recon(x_recon, data)
    return RE_V

def loss_function_stage2(model_output, groundtruth, CUDA, net_type, loss_weights):

    if net_type is 'gru':
        [gru_predicts, encoder_t_embeds, decoder_t_embeds, multi_attention] = model_output
        indexVectors = groundtruth

        # gru loss
        lstm_loss, embed_match_loss = getLSTMloss(gru_predicts, indexVectors, encoder_t_embeds, decoder_t_embeds, CUDA)
        # constraints on the attention weights of gru encode
        ATT = getATTloss(multi_attention, CUDA)

        [weight_stage2_lstm_loss, weight_stage2_emb_loss, weight_stage2_att_loss] = loss_weights

        return [lstm_loss * weight_stage2_lstm_loss, embed_match_loss * weight_stage2_emb_loss, ATT * weight_stage2_att_loss]

    elif net_type is 'nn':
        y_recon = model_output
        ingredients = groundtruth

        RE_T = criterion_recon(y_recon, ingredients)
        return RE_T

def getLSTMloss_v2t(gru_predicts, indexVectors, CUDA):
    lstm_loss = 0
    gru_predicts = gru_predicts.transpose(0, 1)  # (seq, batch, num_word) -> (batch, seq, num_word)
    indexVectors = indexVectors.cpu().numpy()  # word indicators
    total_num = 0

    for i in range(0, gru_predicts.shape[0]):  # for each batch item
        item_seq = indexVectors[i, :]
        seq_len = len(np.where(item_seq > 0)[0])  # compute the index of non-zero entries
        for j in range(0, seq_len):  # for each seq item
            total_num += 1

            GT = torch.zeros([1, gru_predicts.shape[2]])
            GT[0, item_seq[j]-1] = 1
            if CUDA:
                GT = GT.cuda()

            lstm_loss += criterion_lstm(softmax(gru_predicts[i, j, :].unsqueeze(0)), GT)

    return lstm_loss / total_num

def loss_function_stage3(model_output, groundtruth, CUDA, net_type, loss_weights):
    #get individual vectors to compute loss
    [predicts, align_vectors, recon_vectors, cross_vectors] = model_output
    [imgs, indexVectors, ingredients, labels] = groundtruth
    [weight_stage2_lstm_loss, weight_stage2_emb_loss, weight_stage2_att_loss,\
     weight_stage3_cls_loss_v, weight_stage3_cls_loss_t, weight_stage3_align_loss_kl, weight_stage3_align_loss_l2, \
     weight_stage3_recon_loss_v, weight_stage3_v2t_latent, weight_stage3_v2t_lstm] = loss_weights

    labels = labels.cuda()
    #import ipdb; ipdb.set_trace()
    #compute classification losses
    [predicts_v, predicts_t] = predicts
    CE_V = criterion_cls(predicts_v, labels - 1) * weight_stage3_cls_loss_v
    CE_T = criterion_cls(predicts_t, labels - 1) * weight_stage3_cls_loss_t
    
    #compute align losses
    [[x_align1, y_align1], [x_align2, y_align2]] = align_vectors
    AE_kl = criterion_kl(F.log_softmax(x_align1, dim=1), F.softmax(y_align1.detach(), dim=1)) * weight_stage3_align_loss_kl
    AE_l2 = criterion_recon(x_align2, y_align2.detach()) * weight_stage3_align_loss_l2

    #compute recon losses
    [x_recon, y_recon] = recon_vectors
    RE_V = loss_function_stage1(x_recon, imgs) * weight_stage3_recon_loss_v
    if net_type is 'gru':
        RE_T = loss_function_stage2(y_recon, indexVectors, CUDA, net_type,\
                                [weight_stage2_lstm_loss, weight_stage2_emb_loss, weight_stage2_att_loss])
    else:
        RE_T = loss_function_stage2(y_recon, ingredients, CUDA, net_type, \
                                    [weight_stage2_lstm_loss, weight_stage2_emb_loss, weight_stage2_att_loss])

    #compute cross domain ingredient prediction loss
    [[y_latent_recon, y_latent], x2y_ingre_recon] = cross_vectors
    RE_v2t_latent = criterion_recon(y_latent_recon, y_latent.detach()) * weight_stage3_v2t_latent
    if net_type is 'gru':
        RE_v2t_pred = getLSTMloss_v2t(x2y_ingre_recon, indexVectors, CUDA) * weight_stage3_v2t_lstm
    else:
        RE_v2t_pred = criterion_recon(x2y_ingre_recon, ingredients)

    return [CE_V, CE_T, AE_kl, AE_l2, RE_V, RE_T, RE_v2t_latent, RE_v2t_pred]

def compute_loss(model_output, groundtruth, train_stage, CUDA, ingre_net_type, loss_weights):

    if train_stage == 1:
        loss = loss_function_stage1(model_output, groundtruth)
    elif train_stage == 2:
        loss = loss_function_stage2(model_output, groundtruth, CUDA, ingre_net_type, loss_weights)
    elif train_stage == 3:
        loss = loss_function_stage3(model_output, groundtruth, CUDA, ingre_net_type, loss_weights)
    else:
        assert 1 < 0, 'Please input the correct train_stage!'

    return loss