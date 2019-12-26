import scipy.io as matio
import torch
import torch.utils.data
from torch import nn, optim

#load pretrained models for img encoder in stage 1
def get_updateModel(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()

    shared_dict = {k[:7] + k[12:]: v for k, v in pretrained_dict.items() if k[:7] + k[12:] in model_dict}
    cov0_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # import ipdb; ipdb.set_trace()
    model_dict.update(shared_dict)
    model_dict.update(cov0_dict)
    model.load_state_dict(model_dict)

    return model

#load pretrained models in stages 1 and 2
def load_stage_models(model, path, CUDA):
    #load img_model dict
    img_model = torch.load(path[0], map_location='cpu')
    #load ingre model dict
    ingre_model = torch.load(path[1], map_location='cpu')
    #load model dict
    model_dict = model.state_dict()

    #get shared dicts
    if CUDA: #Note the changes in keys, e.g. stage1_model: module.encoder_v.xxx v.s. stage3_model: encoder_v.module.xxx
        shared_dict_v = {k[7:17] + 'module.' + k[17:]: v for k, v in img_model.items() if k[7:17] + 'module.' + k[17:] in model_dict}
        shared_dict_t = {k: v for k, v in ingre_model.items() if k in model_dict}
        #import ipdb; ipdb.set_trace()    
    else: # if in CPU mode
        shared_dict_v = {k: v for k, v in img_model.items() if k in model_dict}
        shared_dict_t = {k: v for k, v in ingre_model.items() if k in model_dict}

    model_dict.update(shared_dict_v)
    model_dict.update(shared_dict_t)
    model.load_state_dict(model_dict)

    return model

# Model
class MyModel_stage1(nn.Module):
    def __init__(self, img_encoder, img_decoder, net_type):
        super(MyModel_stage1, self).__init__()
        # network for image channel
        self.encoder_v = img_encoder
        self.decoder_v = img_decoder

        self.net_type = net_type

        # utilities
        self._initialize_weights()

    def forward(self, x):  # x:image

        #  ipdb.set_trace()
        if self.net_type is 'vgg19bn':
            x_latent = self.encoder_v(x)
            x_recon = self.decoder_v(x_latent)
        else:
            x_latent, a = self.encoder_v(x)
            x_recon = self.decoder_v(x_latent, a)

        return x_recon

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)



class MyModel_stage2(nn.Module):
    def __init__(self, ingre_encoder, ingre_decoder, ingre_net_type):
        super(MyModel_stage2, self).__init__()

        # ingre channel network
        self.encoder_t = ingre_encoder
        self.decoder_t = ingre_decoder

        self.ingre_net_type = ingre_net_type

        self.relu = nn.LeakyReLU()

        self._initialize_weights()

    def forward(self, y):  #y:ingredients
        # compute ingredient vectors
        if self.ingre_net_type is 'gru':
            att_y_latent, encoder_t_embeds, multi_attention = self.encoder_t(y)
            # recon of word embeddings
            gru_predicts, decoder_t_embeds = self.decoder_t(att_y_latent)  # (seq, batch, words)

            return gru_predicts, encoder_t_embeds, decoder_t_embeds, multi_attention

        elif self.ingre_net_type is 'nn':
            y_latent = self.encoder_t(y)
            y_recon = self.decoder_t(y_latent)

            return y_recon

        else:
            assert 1 < 0, 'Please indicate the correct ingre_net_type'


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

# network for nonlinear alignment
class alignNet(nn.Module):
    def __init__(self, hidden_len):
        super(alignNet, self).__init__()

        self.nn1 = nn.Linear(hidden_len, hidden_len)
        self.nn2 = nn.Linear(hidden_len, hidden_len)

        self.relu = nn.ReLU()#nn.LeakyReLU()

    def forward(self, y):
        # compute latent vectors
        y = self.relu(self.nn1(y))
        y = self.relu(self.nn2(y))
        return y

# network for cross-channel v->t mapping
class crossNet(nn.Module):
    def __init__(self, hidden_len):
        super(crossNet, self).__init__()

        self.nn1 = nn.Linear(hidden_len, hidden_len)
        self.nn2 = nn.Linear(hidden_len, hidden_len)
        self.nn3 = nn.Linear(hidden_len, hidden_len)
        self.nn4 = nn.Linear(hidden_len, hidden_len)
        self.nn5 = nn.Linear(hidden_len, hidden_len)
        self.nn6 = nn.Linear(hidden_len, hidden_len)
        self.nn7 = nn.Linear(hidden_len, hidden_len)
        self.nn8 = nn.Linear(hidden_len, hidden_len)

        self.bn1 = nn.BatchNorm1d(hidden_len)
        self.bn2 = nn.BatchNorm1d(hidden_len)
        self.bn3 = nn.BatchNorm1d(hidden_len)
        self.bn4 = nn.BatchNorm1d(hidden_len)
        self.bn5 = nn.BatchNorm1d(hidden_len)
        self.bn6 = nn.BatchNorm1d(hidden_len)
        self.bn7 = nn.BatchNorm1d(hidden_len)

        self.relu = nn.ReLU()#nn.LeakyReLU()

    def forward(self, y):
        # compute latent vectors
        y = self.relu(self.bn1(self.nn1(y)))
        y = self.relu(self.bn2(self.nn2(y)))
        y = self.relu(self.bn3(self.nn3(y)))
        y = self.relu(self.bn4(self.nn4(y)))
        y = self.relu(self.bn5(self.nn5(y)))
        y = self.relu(self.bn6(self.nn6(y)))
        y = self.relu(self.bn7(self.nn7(y)))
        y = self.nn8(y)
        return y

class MyModel_stage3(nn.Module):
    def __init__(self, CUDA, img_encoder, img_decoder, ingre_encoder, ingre_decoder, net_type, latent_len, blk_len, num_class):
        super(MyModel_stage3, self).__init__()

        #network types
        self.net_type = net_type
        self.blk_len = blk_len

        # network for image channel
        self.encoder_v = img_encoder
        self.decoder_v = img_decoder
        if CUDA:
            self.encoder_v = nn.DataParallel(self.encoder_v)
            self.decoder_v = nn.DataParallel(self.decoder_v)

        # ingre channel network
        self.encoder_t = ingre_encoder
        self.decoder_t = ingre_decoder

        # networks for partial heterogeneous transfer
        self.align_x1 = alignNet(blk_len)
        self.align_x2 = alignNet(blk_len)
        self.align_y1 = alignNet(blk_len)
        self.align_y2 = alignNet(blk_len)

        # classifier
        self.classifier = nn.Linear(blk_len, num_class)

        #network for v->t mapping
        self.cross_x = crossNet(latent_len)

        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.relu = nn.LeakyReLU()

        self._initialize_weights()

    def forward(self, x, y):  # x:image, y:ingredient
        #compute image channel outputs
        if self.net_type[0] is 'vgg19bn':
            x_latent = self.encoder_v(x)
            x_recon = self.decoder_v(x_latent)
        else:
            x_latent, a = self.encoder_v(x)
            x_recon = self.decoder_v(x_latent, a)

        #compute ingredient channel outputs
        if self.net_type[1] is 'gru':
            y_latent, encoder_t_embeds, multi_attention = self.encoder_t(y)
            # recon of word embeddings
            gru_predicts, decoder_t_embeds = self.decoder_t(y_latent)  # (seq, batch, words)
        else:
            y_latent = self.encoder_t(y)
            y_recon = self.decoder_t(y_latent)

        # compute v and t predicts in aligned space
        predicts_v, predicts_t, [x_align1, y_align1], [x_align2, y_align2] = self.perform_alignment(x_latent, y_latent)

        #get cross channel v->t mapping
        y_latent_recon = self.cross_x(x_latent)
        if self.net_type[1] is 'gru':
            x2y_ingre_recon, _ = self.decoder_t(y_latent_recon)
            return [predicts_v, predicts_t], [[x_align1, y_align1], [x_align2, y_align2]],\
                   [x_recon, [gru_predicts, encoder_t_embeds, decoder_t_embeds, multi_attention]], [[y_latent_recon, y_latent], x2y_ingre_recon]
        else:
            x2y_ingre_recon = self.decoder_t(y_latent_recon)
            return [predicts_v, predicts_t], [[x_align1, y_align1], [x_align2, y_align2]],\
                   [x_recon, y_recon], [[y_latent_recon, y_latent], x2y_ingre_recon]


    def perform_alignment(self, x_latent, y_latent):
        # compute features in the aligned latent space
        # image channel
        x_align1 = self.align_x1(x_latent[:, :self.blk_len])
        x_align2 = self.align_x2(x_align1)

        # ingre channel
        y_align1 = self.align_y1(y_latent[:, :self.blk_len])
        y_align2 = self.align_y2(y_align1)

        # get predicts
        predicts_v = self.classifier(x_align2)
        predicts_t = self.classifier(y_align2)

        return predicts_v, predicts_t, [x_align1, y_align1], [x_align2, y_align2]


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


def select_img_network(img_net_type, image_size, latent_len):
    if img_net_type is 'resnet50':
        from resnet import resnet50
        img_encoder = resnet50(image_size, pretrained=True)
        from resnet import deresnet50
        img_decoder = deresnet50(image_size, latent_len)
    elif img_net_type is 'vgg19bn':
        from vgg import vgg19_bn
        img_encoder = vgg19_bn(image_size, pretrained=True)
        from vgg import devgg
        img_decoder = devgg(image_size)
    elif img_net_type is 'wrn':
        from wrn import WideResNet
        img_encoder = WideResNet(image_size)
        from resnet import deresnet50
        img_decoder = deresnet50(image_size, latent_len)
    elif img_net_type is 'wiser':
        from wiser import wiser
        img_encoder = wiser()
        from resnet import deresnet50
        img_decoder = deresnet50(image_size, latent_len)
    else:
        assert 1 < 0, 'Please indicate backbone network of image channel with any of resnet50/vgg19bn/wrn/wiser'

    return img_encoder, img_decoder


def select_ingre_network(data_path, CUDA, ingre_net_type, latent_len, max_seq, num_word):
    if ingre_net_type is 'gru':
        # load glove vectors
        gloveVector = matio.loadmat(data_path + 'wordVector.mat')['wordVector']

        from ingre_nets import gru_encoder_t, gru_decoder_t
        ingre_encoder = gru_encoder_t(CUDA, gloveVector, latent_len)
        ingre_decoder = gru_decoder_t(CUDA, gloveVector, latent_len, max_seq, num_word)

    elif ingre_net_type is 'nn':
        from ingre_nets import nn_encoder_t, nn_decoder_t
        ingre_encoder = nn_encoder_t(latent_len, num_word)
        ingre_decoder = nn_decoder_t(latent_len, num_word)

    return ingre_encoder, ingre_decoder


def get_optim(model, learning_rate, opt_w_decay_rate):

    finetune_params_v = [p for k, p in model.named_parameters() if
                         k.startswith('encoder_v.') or
                         k.startswith('decoder_v.')
                         ]

    finetune_params_t = [p for k, p in model.named_parameters() if
                         k.startswith('encoder_t.') or
                         k.startswith('decoder_t.')
                         ]

    normal_params = [p for k, p in model.named_parameters() if
                         not (k.startswith('encoder')
                         or k.startswith('decoder'))
                         ]

    params = [{'params': normal_params}, {'params': finetune_params_v, 'lr': learning_rate[1]}, {'params': finetune_params_t, 'lr': learning_rate[2]}]

    optimizer = optim.Adam(params, weight_decay=opt_w_decay_rate, lr=learning_rate[0])

    return optimizer


def build_mymodel(mode, data_path, CUDA, image_size, latent_len, blk_len, num_class, max_seq, num_word, opt_w_decay_rate, learning_rates, train_stage, net_type, pretrained_img_net_path, stage_model_paths):

    # build model
    if train_stage == 1:
        img_encoder, img_decoder = select_img_network(net_type[0], image_size, latent_len)
        if (net_type[0] is 'wiser') or (net_type[0] is 'wrn'):
            import ipdb; ipdb.set_trace()
            img_encoder = get_updateModel(img_encoder, pretrained_img_net_path)
        model = MyModel_stage1(img_encoder, img_decoder, net_type[0])
        if CUDA:
            model = model.cuda()
            model = nn.DataParallel(model)

        # optimizer
        optimizer = optim.Adam(model.parameters(), weight_decay=opt_w_decay_rate, lr=learning_rates[0])

    elif train_stage == 2:
        ingre_encoder, ingre_decoder = select_ingre_network(data_path, CUDA, net_type[1], latent_len, max_seq, num_word)
        model = MyModel_stage2(ingre_encoder, ingre_decoder, net_type[1])
        if CUDA:
            model = model.cuda()

        # optimizer
        optimizer = optim.Adam(model.parameters(), weight_decay=opt_w_decay_rate, lr=learning_rates[0])

    elif train_stage == 3:
        img_encoder, img_decoder = select_img_network(net_type[0], image_size, latent_len)
        ingre_encoder, ingre_decoder = select_ingre_network(data_path, CUDA, net_type[1], latent_len, max_seq, num_word)
        model = MyModel_stage3(CUDA, img_encoder, img_decoder, ingre_encoder, ingre_decoder, net_type, latent_len, blk_len, num_class)

        if CUDA:
            model = model.cuda()

        if mode is 'train':
            # load pretrained models for image and ingre channels
            model = load_stage_models(model, stage_model_paths, CUDA)
            # optimizer
            optimizer = get_optim(model, learning_rates, opt_w_decay_rate)
        else:
            optimizer = None

    return model, optimizer