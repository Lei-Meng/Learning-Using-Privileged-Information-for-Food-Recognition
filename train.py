import io
import os
import os.path
import time

import torch
import torch.utils.data
from torchvision import transforms





#load environmental settings
import opts
opt = opts.opt_algorithm()



#------Parameter settings-----------------------------------------------------------------------------------------------------------

#file paths
def prepare_intermediate_folders(path):
    if not os.path.exists(path):
        os.makedirs(path)

dataset_indicator = opt.dataset
root_path = opt.root_path
image_path = opt.img_path
data_path = opt.data_path

pretrained_img_net_path = opt.pretrained_img_net_path
stage1_model_path = opt.stage1_model_path
stage1_model_name = 'model_'+ opt.img_net + '.pt'
stage2_model_path = opt.stage2_model_path
stage2_model_name = 'model_' + opt.ingre_net + '.pt'

result_path = opt.result_path

prepare_intermediate_folders(result_path)
prepare_intermediate_folders(stage1_model_path)
prepare_intermediate_folders(stage2_model_path)

#dataset info
num_class = opt.dataset_num_class
max_seq = opt.dataset_max_seq
num_word = opt.dataset_num_ingre
avg_word = opt.dataset_avg_ingre

# Image Info
no_of_channels = 3
image_size = [256, 256]  # [64,64]

#train type
train_stage = opt.stage
mode = opt.mode
img_net_type = opt.img_net
ingre_net_type = opt.ingre_net


#algorithm hyperparameter - changed configuration to this instead of argparse for easier interaction
CUDA = 1  # 1 for True; 0 for False
SEED = 1
BATCH_SIZE = 64
LOG_INTERVAL = 1
learning_rate = 1e-3
learning_rate_finetune_v = 1e-6
learning_rate_finetune_t = 1e-6
decay_rate = 0.1

if img_net_type is 'vgg19bn':
    latent_len = 4096
else:
    latent_len = 2048
blk_len = int(latent_len * 3 / 8)

weight_stage2_lstm_loss = 1e2
weight_stage2_emb_loss = 1e2
weight_stage2_att_loss = 1e4

weight_stage3_cls_loss_v = 2e1
weight_stage3_cls_loss_t = 5e0
weight_stage3_align_loss_kl = 1e4
weight_stage3_align_loss_l2 = 1e1
weight_stage3_recon_loss_v = 1e0
weight_stage3_v2t_latent = 5e0
weight_stage3_v2t_ingre_pred = 1e1

opt_w_decay_rate = 1e-3

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
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

#create dataset
from build_dataset import build_dataset
dataset = build_dataset(train_stage, image_path, data_path, transform_img, mode, dataset_indicator)

#dataloader
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)



#--Model----------------------------------------------------------------------------------

from mymodel import build_mymodel
model, optimizer = build_mymodel(mode, data_path, CUDA, image_size, latent_len, blk_len, num_class, max_seq, num_word,\
                                 opt_w_decay_rate, [learning_rate, learning_rate_finetune_v, learning_rate_finetune_t], train_stage, [img_net_type, ingre_net_type],\
                                 pretrained_img_net_path, [stage1_model_path+stage1_model_name, stage2_model_path+stage2_model_name])
model.train()

# --Loss & performance----------------------------------------------------------------

# loss
from loss import compute_loss
# performance
from compute_performance import compute_performance


# --train----------------------------------------------------------------
def print_state(state):
    print(state)
    # records current progress for tracking purpose
    with io.open(result_path + 'model_batch_train_loss.txt', 'a', encoding='utf-8') as file:
        file.write(state + '\n\n')

def train_epoch(epoch, decay, train_stage):

    print('Training starts..')

    train_loss = 0

    if train_stage == 3:
        top1_accuracy_total_V = 0
        top5_accuracy_total_V = 0
        top1_accuracy_total_T = 0
        top5_accuracy_total_T = 0

        ingre_pred_precision_total = 0

    total_time = time.time()

    for batch_idx, (data) in enumerate(train_loader):

        if batch_idx == 2:
            break

        start_time = time.time()

        # load data
        if train_stage == 1:
            if CUDA:
                data = data.cuda()
        elif train_stage == 2:
            [indexVectors, ingredients] = data
            if CUDA:
                indexVectors = indexVectors.cuda()
                ingredients = ingredients.cuda()
        elif train_stage == 3:
            [imgs, indexVectors, ingredients, labels] = data
            if CUDA:
                imgs = imgs.cuda()
                indexVectors = indexVectors.cuda()
                ingredients = ingredients.cuda()
        else:
            assert 1 < 0, 'Please fill train_stage!'


        # perform model inference & compute loss
        if train_stage == 1:
            #perform prediction
            x_recon = model(data)
            #compute loss
            RE_V = compute_loss(x_recon, data, train_stage, CUDA, None, None)
            final_loss = RE_V

        elif train_stage == 2:
            if ingre_net_type is 'gru':
                #perform prediction
                if batch_idx == 6:
                    a=1
                [gru_predicts, encoder_t_embeds, decoder_t_embeds, multi_attention] = model(indexVectors)
                #compute loss
                [lstm_loss, embed_match_loss, ATT] =\
                    compute_loss([gru_predicts, encoder_t_embeds, decoder_t_embeds, multi_attention], indexVectors,\
                                 train_stage, CUDA, ingre_net_type,\
                                 [weight_stage2_lstm_loss, weight_stage2_emb_loss, weight_stage2_att_loss])
                final_loss = lstm_loss + embed_match_loss + ATT

            elif ingre_net_type is 'nn':
                # perform prediction
                y_recon = model(ingredients)
                # compute loss
                RE_T = compute_loss(y_recon, ingredients, train_stage, CUDA, ingre_net_type, None)
                final_loss = RE_T

            else:
                assert 1 < 0, 'Please indicate the correct ingre_net_type!'

        elif train_stage == 3:
            #perform prediction
            if ingre_net_type is 'gru':
                predicts, align_vectors, recon_vectors, cross_vectors = model(imgs, indexVectors)
            else:
                predicts, align_vectors, recon_vectors, cross_vectors = model(imgs, ingredients)
            #compute loss
            [CE_V, CE_T, AE_kl, AE_l2, RE_V, RE_T, RE_v2t_latent, RE_v2t_ingre_pred] =\
                compute_loss([predicts, align_vectors, recon_vectors, cross_vectors], [imgs, indexVectors, ingredients, labels], train_stage, CUDA, ingre_net_type,\
                             [weight_stage2_lstm_loss, weight_stage2_emb_loss, weight_stage2_att_loss,\
                              weight_stage3_cls_loss_v, weight_stage3_cls_loss_t, weight_stage3_align_loss_kl, weight_stage3_align_loss_l2,\
                              weight_stage3_recon_loss_v, weight_stage3_v2t_latent, weight_stage3_v2t_ingre_pred])

            final_loss = CE_V + CE_T + AE_kl + AE_l2 + RE_V + RE_v2t_latent + RE_v2t_ingre_pred
            if ingre_net_type is 'gru':
                [lstm_loss, embed_match_loss, ATT] = RE_T
                final_loss += lstm_loss + embed_match_loss + ATT
            elif ingre_net_type is 'nn':
                final_loss += RE_T
            else:
                assert 1 < 0, 'Please indicate the correct ingre_net_type!'

        else:
            assert 1 < 0, 'Please input the correct train_stage!'


        # optimization for myModel
        optimizer.zero_grad()
        final_loss.backward()
        train_loss += final_loss.data
        optimizer.step()


        # compute performance
        if train_stage == 3:
            x2y_ingre_recon = cross_vectors[1]
            [top1_hit_V, top5_hit_V], [top1_hit_T, top5_hit_T], ingre_precision =\
                compute_performance(mode, predicts, x2y_ingre_recon, labels, indexVectors, ingredients, ingre_net_type, avg_word, None, None, None)

            # top 1 accuracy
            top1_accuracy_total_V += top1_hit_V
            top1_accuracy_cur_V = top1_hit_V / float(len(labels))

            top1_accuracy_total_T += top1_hit_T
            top1_accuracy_cur_T = top1_hit_T / float(len(labels))

            # top 5 accuracy
            top5_accuracy_total_V += top5_hit_V
            top5_accuracy_cur_V = top5_hit_V / float(len(labels))

            top5_accuracy_total_T += top5_hit_T
            top5_accuracy_cur_T = top5_hit_T / float(len(labels))

            #ingredient prediction performance
            ingre_pred_precision_total += ingre_precision

        if epoch == 1 and batch_idx == 0:
            if train_stage == 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] | RE_V: {:.4f} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                    RE_V,
                    round((time.time() - start_time), 4),
                    round((time.time() - total_time), 4)))
            elif train_stage == 2:
                if ingre_net_type is 'gru':
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] | lstm_loss: {:.4f} | embed_match_loss: {:.4f} | ATT: {:.4f} | Time:{} | Total_Time:{}'.format(
                            epoch, (batch_idx + 1) * len(ingredients), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                            lstm_loss, embed_match_loss, ATT,
                            round((time.time() - start_time), 4),
                            round((time.time() - total_time), 4)))
                elif ingre_net_type is 'nn':
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] | RE_T: {:.4f} | Time:{} | Total_Time:{}'.format(
                        epoch, (batch_idx + 1) * len(ingredients), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                        RE_T,
                        round((time.time() - start_time), 4),
                        round((time.time() - total_time), 4)))
            elif train_stage == 3:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] | CE_V: {:.4f} | CE_T: {:.4f} | AE_kl: {:.4f} | AE_l2: {:.4f} | RE_l: {:.4f} | RE_p: {:.4f} | Acc_v: {:.4f} | Acc_t: {:.4f} | Acc_ingre: {:.4f} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(ingredients), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                    CE_V, CE_T, AE_kl, AE_l2, RE_v2t_latent, RE_v2t_ingre_pred,
                    top1_accuracy_cur_V, top1_accuracy_cur_T, ingre_precision,
                    round((time.time() - start_time), 4),
                    round((time.time() - total_time), 4)))

            with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
                file.write('Epoach {}: | Avg loss: {:.4f}\n'.format(epoch, train_loss))

        elif batch_idx % LOG_INTERVAL == 0:
            if train_stage == 1:
                state = 'Train Epoch: {} [{}/{} ({:.0f}%)] | RE_V: {:.4f} | Time:{} | Total_Time:{}'.format(
                        epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                        RE_V,
                        round((time.time() - start_time), 4) * LOG_INTERVAL,
                        round((time.time() - total_time), 4))
                print_state(state)

            elif train_stage == 2:
                if ingre_net_type is 'gru':
                    state = 'Train Epoch: {} [{}/{} ({:.0f}%)] | lstm_loss: {:.4f} | embed_match_loss: {:.4f} | ATT: {:.4f} | Time:{} | Total_Time:{}'.format(
                            epoch, (batch_idx + 1) * len(ingredients), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                            lstm_loss.data, embed_match_loss.data, ATT.data,
                            round((time.time() - start_time), 4) * LOG_INTERVAL,
                            round((time.time() - total_time), 4))
                    print_state(state)

                elif ingre_net_type is 'nn':
                    state = 'Train Epoch: {} [{}/{} ({:.0f}%)] | RE_T: {:.4f} | Time:{} | Total_Time:{}'.format(
                        epoch, (batch_idx + 1) * len(ingredients), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                        RE_T,
                        round((time.time() - start_time), 4) * LOG_INTERVAL,
                        round((time.time() - total_time), 4))
                    print_state(state)
            elif train_stage == 3:
                state = 'Train Epoch: {} [{}/{} ({:.0f}%)] | CE_V: {:.4f} | CE_T: {:.4f} | AE_kl: {:.4f} | AE_l2: {:.4f} | RE_l: {:.4f} | RE_p: {:.4f} | Acc_v: {:.4f} | Acc_t: {:.4f} | Acc_ingre: {:.4f} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(ingredients), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                    CE_V, CE_T, AE_kl, AE_l2, RE_v2t_latent, RE_v2t_ingre_pred,
                    top1_accuracy_cur_V, top1_accuracy_cur_T, ingre_precision,
                    round((time.time() - start_time), 4) * LOG_INTERVAL,
                    round((time.time() - total_time), 4))
                print_state(state)

    #record epoch-level results
    if train_stage != 3:
        print('====> Epoch: {} | Avg loss: {:.4f} | Time:{}\n'.format(epoch, train_loss / len(train_loader), round((time.time() - total_time), 4)))
        with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
            file.write('Epoach {}: | Avg loss: {:.4f}\n'.format(epoch, train_loss / len(train_loader)))
    else:
        print('====> Epoch: {} | Avg loss: {:.4f} | Top1_Acc_V: {:.4f} | Top5_Acc_V: {:.4f} | Top1_Acc_T: {:.4f} | Top5_Acc_T: {:.4f} | Ingre_pred_Acc: {:.4f} | Time:{}\n'.format(
            epoch, train_loss / len(train_loader),
            top1_accuracy_total_V / len(train_loader.dataset), top5_accuracy_total_V / len(train_loader.dataset),
            top1_accuracy_total_T / len(train_loader.dataset), top5_accuracy_total_T / len(train_loader.dataset),
            ingre_pred_precision_total / len(train_loader),
            round((time.time() - total_time), 4)))
        with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
            file.write('Epoach {}: Avg loss: {:.4f} | Top1_Acc_V: {:.4f} | Top5_Acc_V: {:.4f} | Top1_Acc_T: {:.4f} | Top5_Acc_T: {:.4f} | Ingre_pred_Acc: {:.4f}\n'.format(epoch, train_loss / len(train_loader),
                                                top1_accuracy_total_V / len(train_loader.dataset),
                                                top5_accuracy_total_V / len(train_loader.dataset),
                                                top1_accuracy_total_T / len(train_loader.dataset),
                                                top5_accuracy_total_T / len(train_loader.dataset),
                                                ingre_pred_precision_total / len(train_loader)))

    # save current model
    if epoch > decay:
        torch.save(model.state_dict(), result_path + 'model' + str(epoch) + '.pt')

lr_names = ['lr', 'lr_finetune_v', 'lr_finetune_t']
def lr_scheduler(epoch, lr_decay_iter, decay_rate):
    if not (epoch % lr_decay_iter):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['lr'] * decay_rate
            print(lr_names[i] + ' : {}'.format(optimizer.param_groups[i]['lr']))
    else:
        for i in range(len(optimizer.param_groups)):
            print(lr_names[i] + ' : {}'.format(optimizer.param_groups[i]['lr']))        
            
            
# excute main
for epoch in range(1, EPOCHS + 1):
    lr_scheduler(epoch, lr_decay, decay_rate)
    train_epoch(epoch, lr_decay, train_stage)





