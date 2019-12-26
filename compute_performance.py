import numpy as np
import io




def compute_cls_precision(predicts, labels):
    sorted_predicts = predicts.detach().cpu().numpy().argsort()
    top1_labels = sorted_predicts[:, -1:][:, 0]
    match = float(sum(top1_labels - labels + 1 == 0))

    top5_labels = sorted_predicts[:, -5:]
    hit = 0
    for i in range(0, labels.size(0)):
        hit += (labels[i] - 1) in top5_labels[i, :]

    return match, hit


def compute_ingre_pred_precision_nn(predicts, labels, avg_word):
    # get top-n predicts
    predicts = predicts.detach().cpu().numpy()
    sorted_predicts = (-predicts).argsort()
    top_n_inds = sorted_predicts[:, :avg_word]

    labels = labels.detach().cpu().numpy()

    # compute top-n hits for each sample
    hit = np.zeros([len(labels), avg_word])
    for i in range(len(labels)):  # for each sample \in [0,batch_size]
        for j in range(1, avg_word + 1):  # for each value of n \in [1,5]
            for k in range(j):  # for each rank of j
                if labels[i, top_n_inds[i, k]] - 1 == 0:
                    hit[i, j - 1] += 1  # j-1 since hit is 0-indexed

    # compute precision
    # get denominator
    denominator = np.arange(avg_word) + 1
    denominator = np.tile(denominator, [len(labels), 1])
    # get precision
    precision = hit / denominator

    # compute recall
    # get denominator
    denominator = np.sum(labels, axis=1)
    denominator = np.tile(np.expand_dims(denominator, axis=1), [1, avg_word])
    # get recall
    recall = hit / denominator

    return np.mean(precision, 0), np.mean(recall, 0)


def compute_ingre_pred_precision_gru(mode, x2y_ingre_recon, indexVectors, labels, class_names, ingre_name, result_path):
    #x2y_ingre_recon: (seq, batch, number_of_ingredients), indexVectors: (seq, batch, number_of_ingredients)

    #initialize intermediate variables
    seq_len = x2y_ingre_recon.shape[0] #the length of input sequence
    batch_size = x2y_ingre_recon.shape[1]
    num_ingre = x2y_ingre_recon.shape[2]

    PredictIndex = np.zeros(seq_len) # record the index of the top-1 predicted word for each entry of seq
    precisionCount = np.zeros(seq_len)  # count the hit of correct predicts for precision in PredictIndex, i.e. top1 predicted words
    num_word_per_location = np.zeros(seq_len) # to facilitate avg precision considering seq lens
    batch_sum_precision = np.zeros(seq_len) # store the performance in seq-level precision of top-n positions

    mode_flag = not (mode is 'train') # if in val or test mode

    if mode_flag:#include computation for seq-level recall and word-level precision
        #variables for seq-level recall
        batch_sum_recall = np.zeros(seq_len)  # count the hit of correct predicts for recall in PredictIndex, i.e. top1 predicted words

        # variables for word-level precision
        Top_n_word = 5
        precisionCount_word = np.zeros(Top_n_word)  # record the number of correct predicts in top 1-5, at word level
        batch_sum_precision_word = np.zeros(Top_n_word)  # record the number of correct predicts in top 1-5, at word level, for the batch

    #compute seq-level precision in top-n positions as evaluation metric
    x2y_ingre_recon = x2y_ingre_recon.transpose(0, 1) # make it of (batch, seq, number_of_ingredients)
    x2y_ingre_recon = x2y_ingre_recon.cpu().data.numpy()
    indexVectors = indexVectors.cpu().data.numpy() # (batch, seq)

    for i in range(0, batch_size):  # for each batch item

        #get groundtruth seq, e.g. [1,3,12,3,67,88]
        item_seq = indexVectors[i, :]

        # to avoid possible testing data with zero entries
        if np.sum(item_seq) == 0:
            print('no input at data {}'.format(i))
            continue

        #clear information from last batch item
        PredictIndex[:] = 0
        precisionCount[:] = 0

        #get groundtruth information
        item_seq_len = len(np.where(item_seq > 0)[0])

        #update word counts in seq locations in batch
        num_word_per_location[:item_seq_len] += 1

        # if not in train mode, record detailed val/test results
        if mode_flag:
            with io.open(result_path + 'img2tag.txt', 'a', encoding='utf-8') as file:
                file.write('Class: {}\n'.format(class_names[labels[i]]))
                file.write('True Tags: ')
                for p in range(0, item_seq_len):
                    if p == 0: line = '{},'
                    elif p == item_seq_len - 1: line = ' {}'
                    else: line = ' {},'
                    file.write(line.format(ingre_name[item_seq[p]-1]))
                # for each seq, record the ranks of labels in the predicts in the next a few lines
                file.write('\n')
                file.write('Rank in Predicts: ')

        #get top-1 predicted ingredient for each entry in sequence
        for j in range(0, item_seq_len):  # for each seq word
            sorted_predicts = x2y_ingre_recon[i, j,:].argsort()  # ranked list of indexes, value:low -> high
            PredictIndex[j] = sorted_predicts[-1]
            #import ipdb; ipdb.set_trace()
            #check the match of prediction
            if PredictIndex[j] - item_seq[j] + 1 == 0:
                precisionCount[j:item_seq_len] += 1

            if mode_flag:
                # compute position of groundtruth word in prediction
                position_in_list = np.where(sorted_predicts - item_seq[j] + 1 == 0)[0][0]
                rank = num_ingre - 1 - position_in_list

                # count the top-n hit for individual words
                if rank < Top_n_word:
                    precisionCount_word[rank:Top_n_word] += 1

                # record the predict details
                with io.open(result_path + 'img2tag.txt', 'a', encoding='utf-8') as file:
                    if j == 0: line = '{}:{},'
                    elif j == item_seq_len - 1: line = ' {}:{}'
                    else: line = ' {}:{},'
                    file.write(line.format(ingre_name[item_seq[j]-1], rank))

        #record seq-level precision
        avg_precision = precisionCount / (np.arange(seq_len) + 1)
        batch_sum_precision += avg_precision

        #record seq-level recall and word-level precision
        if mode_flag:
            avg_recall = precisionCount / item_seq_len
            batch_sum_recall += avg_recall

            avg_precision_word = precisionCount_word / item_seq_len
            batch_sum_precision_word += avg_precision_word

            #record the word-level avg precision
            with io.open(result_path + 'img2tag.txt', 'a', encoding='utf-8') as file:
                file.write('\n')
                file.write('Top-{} Avg Word Precision: {}'.format(Top_n_word, avg_precision_word))
                file.write('\n')
                file.write('Avg Precision: {}'.format(avg_precision))
                file.write('\n')
                file.write('Avg Recall: {}'.format(avg_recall))
                file.write('\n')
                file.write('All predicted words: {}\n\n'.format(ingre_name[PredictIndex[:item_seq_len].astype(np.int)]))

    non_zero_entries = np.where(num_word_per_location > 0)[0]
    batch_sum_precision[non_zero_entries] = batch_sum_precision[non_zero_entries] / num_word_per_location[non_zero_entries]

    if not mode_flag:
        return [batch_sum_precision]
    else:
        batch_sum_recall[non_zero_entries] = batch_sum_recall[non_zero_entries] / num_word_per_location[non_zero_entries]
        batch_sum_precision_word = batch_sum_precision_word / batch_size
        return [batch_sum_precision, batch_sum_recall, batch_sum_precision_word]


def compute_ingre_prediction_performance(mode, x2y_ingre_recon, indexVectors, ingredients, ingre_net_type, avg_word, labels, class_names, ingre_word_names, result_path):
    if mode is 'train':
        if ingre_net_type is 'gru':
            [avg_precision] = compute_ingre_pred_precision_gru(mode, x2y_ingre_recon, indexVectors, None, None, None, None)
        else:
            avg_precision, _ = compute_ingre_pred_precision_nn(x2y_ingre_recon, ingredients, avg_word)
        return avg_precision[0]

    else: #if test & val
        if ingre_net_type is 'gru':# packed performance: [avg_precision, avg_recall, avg_precision_word]
            ingre_pred_performance = compute_ingre_pred_precision_gru(mode, x2y_ingre_recon, indexVectors, labels, class_names, ingre_word_names, result_path)
        else:# packed performance: [avg_precision, avg_recall]
            ingre_pred_performance = compute_ingre_pred_precision_nn(x2y_ingre_recon, ingredients, avg_word)
        return ingre_pred_performance

def compute_performance(mode, predicts, x2y_ingre_recon, labels, indexVectors, ingredients, ingre_net_type, avg_word, class_names, ingre_word_names, result_path):

    [predicts_v, predicts_t] = predicts

    #compute performance of image and ingredient features
    top1_hit_V, top5_hit_V = compute_cls_precision(predicts_v, labels)
    top1_hit_T, top5_hit_T = compute_cls_precision(predicts_t, labels)

    #compute performance of ingredient prediction using image features
    ingre_pred_performance = compute_ingre_prediction_performance(mode, x2y_ingre_recon, indexVectors, ingredients, ingre_net_type, avg_word, labels, class_names, ingre_word_names, result_path)

    return [top1_hit_V, top5_hit_V], [top1_hit_T, top5_hit_T], ingre_pred_performance