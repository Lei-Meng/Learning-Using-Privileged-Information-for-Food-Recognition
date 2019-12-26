import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


#---ResNet encoder------------------------------------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, image_size):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # define resnet encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0])  # 64-64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 64-128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 128-256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 256-512
        self.avgpooling = nn.AvgPool2d(image_size[0] // (2 ** 5), stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # x:image y:ingredient

        x = self.conv1(x)  # ／2
        x = self.bn1(x)
        x = self.relu(x)
        [x, a] = self.maxpool(x)  # ／2
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)  # ／2
        # print(x.shape)
        x = self.layer3(x)  # ／2
        # print(x.shape)
        x = self.layer4(x)  # ／2
        # print(x.shape)
        x = self.avgpooling(x)  # (1x1)
        # print(x.shape)
        x = x.view(x.shape[0], x.shape[1])

        return x, a


def resnet50(image_size, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], image_size, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()

        shared_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(shared_dict)
        model.load_state_dict(model_dict)
    return model


#---ResNet decoder------------------------------------------------------------------------------------------------------
def Deconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    if (stride - 2) == 0:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, output_padding=1, bias=False)
    else:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)


class DeBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DeBasicBlock, self).__init__()
        self.conv1 = Deconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = Deconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def Deconv_Bottleneck(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    if (stride - 2) == 0:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, output_padding=1, bias=False)
    else:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)


class DeBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, outplanes, stride=1, downsample=None):
        super(DeBottleneck, self).__init__()
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Deconv_Bottleneck(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.ConvTranspose2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class deResNet(nn.Module):
    def __init__(self, block, layers, image_size, latent_len):
        self.inplanes = latent_len
        super(deResNet, self).__init__()

        # define resnet decoder
        self.image_size = image_size
        self.latent_len = latent_len
        self.unavgpool = nn.Conv2d(latent_len, latent_len, kernel_size=1, stride=1)
        self.layer5 = self._make_Delayer(block, 512, layers[3], stride=2)  # 512-256
        self.layer6 = self._make_Delayer(block, 256, layers[2], stride=2)  # 256-128
        self.layer7 = self._make_Delayer(block, 128, layers[1], stride=2)  # 128-64
        self.layer8 = self._make_Delayer(block, 64, layers[0], stride=1)  # 64-64
        self.unmaxpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.deconv9 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_Delayer(self, block, planes, blocks, stride=1):
        layers = []
        for i in range(0, blocks - 1):
            layers.append(block(self.inplanes, planes, self.inplanes))

        if self.inplanes == 256:
            downsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, self.inplanes // 4,
                                   kernel_size=1, stride=stride, output_padding=0, bias=False),
                nn.BatchNorm2d(self.inplanes // 4),
            )

            layers.append(block(self.inplanes, planes, self.inplanes // 4, stride, downsample))
        else:
            downsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, self.inplanes // 2,
                                   kernel_size=1, stride=stride, output_padding=1, bias=False),
                nn.BatchNorm2d(self.inplanes // 2),
            )

            layers.append(block(self.inplanes, planes, self.inplanes // 2, stride, downsample))

        self.inplanes = self.inplanes // 2

        return nn.Sequential(*layers)

    def forward(self, x_latent, a):
        x = x_latent.view(x_latent.shape[0], self.latent_len, 1, 1)
        # print(x.shape)
        # print(x.shape)
        x = F.upsample(x, scale_factor=self.image_size[0] // (2 ** 5), mode='nearest')
        x = self.unavgpool(x)
        x = self.layer5(x)
        # print(x.shape)
        x = self.layer6(x)
        # print(x.shape)
        x = self.layer7(x)
        # print(x.shape)
        x = self.layer8(x)
        # print(x.shape)
        x = self.unmaxpool(x, a, output_size=[x.shape[0], x.shape[1], self.image_size[0] // 2, self.image_size[0] // 2])
        # print(x.shape)
        x = self.deconv9(x)
        # print(x.shape)
        x = self.sigmoid(x)
        return x


def deresnet50(image_size, latent_len, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = deResNet(DeBottleneck, [3, 4, 6, 3], image_size, latent_len, **kwargs)
    # if pretrained:
    #     pretrained_dict = model_zoo.load_url(xxx)
    #     model_dict = model.state_dict()
    #
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)

    return model