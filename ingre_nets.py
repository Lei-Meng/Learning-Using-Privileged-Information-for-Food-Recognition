import numpy as np


import torch
import torch.utils.data
from torch import nn
from torch.nn import Parameter

#---GRU network------------------------------------------------------------------------------------------------------

#encoder
class gru_encoder_t(nn.Module):
    def __init__(self, CUDA, gloveVector, num_hidden, num_key_ingre=5):
        super(gru_encoder_t, self).__init__()

        #initialize word embeddings using gloveVector
            #add a zero vector on top of gloveVector for padding_idx
        wordVector = np.concatenate([np.zeros((1, gloveVector.shape[1])), gloveVector], 0)
            #initialization
        self.embedding = nn.Embedding(wordVector.shape[0], wordVector.shape[1], padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(wordVector))

        self.CUDA = CUDA
        self.num_hidden = num_hidden
        self.num_key_ingre = num_key_ingre

        self.gru = nn.GRU(wordVector.shape[1], self.num_hidden)

        # linear layers for gru attention
        self.ws1 = nn.Linear(num_hidden, num_hidden)
        self.ws2 = nn.Linear(num_hidden, num_key_ingre)
        self.ingre2att = nn.Linear(num_key_ingre, 1)
        self.relu = nn.LeakyReLU()

        self._initialize_weights()

    def forward(self, y):
        # compute latent vectors
        encoder_t_embeds = self.embedding(y)
        encoder_t_embeds = encoder_t_embeds.permute(1, 0, 2)

        # obtain gru output of hidden vectors
        h0_en = Parameter(torch.zeros((1, y.shape[0], self.num_hidden), requires_grad=True))
        if self.CUDA:
            h0_en = h0_en.cuda()
        y_embeds, _ = self.gru(encoder_t_embeds, h0_en)

        att_y_embeds, multi_attention = self.getAttention(y_embeds)

        return att_y_embeds, encoder_t_embeds, multi_attention

    def getAttention(self, y_embeds):
        # y_embeds = self.dropout(y_embeds)  # (seq, batch, hidden)
        y_embeds = y_embeds.transpose(0, 1)  # (batch, seq, hidden)
        # compute multi-focus self attention by a two-layer mapping
        # (batch, seq, hidden) -> (batch, seq, hidden) -> (batch, seq, self.num_key_ingre)
        multi_attention = self.ws2(self.relu(self.ws1(y_embeds)))
        # compute attended embeddings in terms of focus
        multi_attention = multi_attention.transpose(1, 2)  # (batch, self.num_key_ingre, seq)
        att_y_embeds = multi_attention.bmm(y_embeds)  # (batch, self.num_key_ingre, hidden)
        # compute the aggregated hidden vector
        att_y_embeds = self.ingre2att(att_y_embeds.transpose(1, 2)).squeeze(2)  # (batch, hidden)
        return att_y_embeds, multi_attention.transpose(1, 2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


#decoder
class gru_decoder_t(nn.Module):
    def __init__(self, CUDA, gloveVector, num_hidden, max_seq, num_word, num_glove=300):
        super(gru_decoder_t, self).__init__()

        #initialize word embeddings using gloveVector
            #add a zero vector on top of gloveVector for padding_idx
        wordVector = np.concatenate([np.zeros((1, gloveVector.shape[1])), gloveVector], 0)
            #initialization
        self.embedding = nn.Embedding(wordVector.shape[0], wordVector.shape[1], padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(wordVector))

        self.CUDA = CUDA
        self.num_glove = num_glove
        self.max_seq = max_seq



        # self.register_parameter('h0_de', None)
        self.hiddenMap1 = nn.Linear(num_hidden + num_glove, num_hidden)
        self.hiddenMap2 = nn.Linear(num_hidden, num_hidden)

        self.gru = nn.GRU(num_hidden, num_glove)  # nn.GRU(num_hidden, num_glove, dropout=0.1)

        self.wordpredict1 = nn.Linear(num_glove, self.num_glove)
        self.wordpredict2 = nn.Linear(num_glove, num_word)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()

        self._initialize_weights()

    def forward(self, y):
        # Use latent_y = (batch, num_hidden) as input to predict a sequence of ingredient words
        # y has size (batch,num_hidden)

        h_de = []  # store the output hidden vectors of gru
        gru_predicts = []  # store the predicts of gru for words

        h0_de = Parameter(torch.zeros((1, y.shape[0], self.num_glove), requires_grad=True))
        padding = torch.zeros(y.shape[0], self.num_glove)
        if self.CUDA:
            h0_de = h0_de.cuda()
            padding = padding.cuda()

        current_input = torch.cat([y, padding], 1).unsqueeze(0)  # (1, batch, num_hidden+num_glove)
        current_input = self.hiddenMap2(self.relu(self.hiddenMap1(current_input)))

        prev_hidden = h0_de

        for i in range(0, self.max_seq):  # for each of the max_seq for decoder
            # NOTE: current_hidden = prev_hidden, we use different notations to clarify their roles
            current_hidden, prev_hidden = self.gruLoop(current_input, prev_hidden)
            # save gru output
            h_de.append(current_hidden)
            # compute next input to gru, the glove embedding vector of the current predicted word
            current_input, wordPredicts = self.getNextInput(y, current_hidden)

            gru_predicts.append(wordPredicts)

        return torch.cat(gru_predicts, 0), torch.cat(h_de, 0)  # make it a tensor (seq, batch, num_word)

    def getNextInput(self, y, current_hidden):
        # get embedding of the predicted words
        wordPredicts = self.wordpredict2(self.relu(self.wordpredict1(current_hidden))).squeeze(
            0)  # (1, batch, num_glove) -> (batch, num_word)
        wordIndex = torch.argmax(wordPredicts, dim=1)  # (batch, 1)
        embeddings = self.embedding(
            wordIndex + 1)  # (batch,num_glove) note that the index 0 of Embedding is for non-entry
        # fuse the embedding with y_latent using a non-linear mapping to extract sufficient information for the next word
        next_input = self.hiddenMap2(self.relu(self.hiddenMap1(torch.cat([y, embeddings], 1)))).unsqueeze(0)

        return next_input, wordPredicts.unsqueeze(0)

    def gruLoop(self, current_input, prev_hidden):
        # use it to avoid a modification of prev_hidden with inplace operation
        output, hidden = self.gru(current_input, prev_hidden)
        return output, hidden

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)







#---nn network------------------------------------------------------------------------------------------------------

class nn_encoder_t(nn.Module):
    def __init__(self, latent_len, num_ingre_feature):
        super(nn_encoder_t, self).__init__()

        # define ingredient encoder for 353 input features

        self.nn1 = nn.Linear(num_ingre_feature, num_ingre_feature)
        self.nn2 = nn.Linear(num_ingre_feature, latent_len)

        self.relu = nn.ReLU()#nn.leakyReLU()

    def forward(self, y):
        # compute latent vectors
        y = self.relu(self.nn1(y))
        y = self.nn2(y)

        return y


class nn_decoder_t(nn.Module):
    def __init__(self, latent_len, num_ingre_feature):
        super(nn_decoder_t, self).__init__()

        # define ingredient decoder
        self.nn3 = nn.Linear(latent_len, num_ingre_feature)
        self.nn4 = nn.Linear(num_ingre_feature, num_ingre_feature)

        self.relu = nn.ReLU()#nn.leakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y):
        # compute latent vectors
        y = self.relu(self.nn3(y))
        y = self.sigmoid(self.nn4(y))

        return y