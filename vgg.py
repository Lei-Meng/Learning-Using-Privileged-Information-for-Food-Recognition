import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import re

model_urls = {
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


#---vgg encoder------------------------------------------------------------------------------------------------------

class VGG(nn.Module):
    def __init__(self, features, image_size, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * ((image_size[0] // (2 ** 5)) ** 2), 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg19_bn(image_size, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False

    model = VGG(make_layers(cfg['E'], batch_norm=True), image_size, **kwargs)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict
                           and not re.match(k, 'classifier.0.weight')
                           and not re.match(k, 'classifier.6.weight')
                           and not re.match(k, 'classifier.6.bias')
                           }

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


#---vgg decoder------------------------------------------------------------------------------------------------------
class SegNet(nn.Module):
    def __init__(self, image_size, init_weights=True):
        super(SegNet, self).__init__()

        self.image_size = image_size

        self.latent_re = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.Linear(4096, 512 * ((image_size[0] // (2 ** 5)) ** 2)),
        )

        batchNorm_momentum = 0.1

        self.upsample5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv54d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn54d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.upsample4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv44d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn44d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.upsample3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv34d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn34d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x5p = self.latent_re(x)
        x5p = x5p.view(x.size(0), 512, self.image_size[0] // (2 ** 5), self.image_size[0] // (2 ** 5))
        #print(x5p.shape)
        # Stage 5d
        x5d = self.upsample5(x5p)
        #print(x5d.shape)
        x54d = F.relu(self.bn54d(self.conv54d(x5d)))
        x53d = F.relu(self.bn53d(self.conv53d(x54d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = self.upsample4(x51d)
        #print(x4d.shape)
        x44d = F.relu(self.bn44d(self.conv44d(x4d)))
        x43d = F.relu(self.bn43d(self.conv43d(x44d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = self.upsample3(x41d)
        #print(x3d.shape)
        x34d = F.relu(self.bn34d(self.conv34d(x3d)))
        x33d = F.relu(self.bn33d(self.conv33d(x34d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = self.upsample2(x31d)
        #print(x2d.shape)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = self.upsample1(x21d)
        #print(x1d.shape)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        x_recon = self.sigmoid(x11d)
        return x_recon

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def devgg(image_size, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False

    model = SegNet(image_size, **kwargs)

    return model