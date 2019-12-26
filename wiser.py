import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, inplane, out_plane, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        inter_plane = out_plane // 2
        self.conv0 = nn.Conv2d(inplane, inter_plane, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(inter_plane, inter_plane, 3, stride, 1)
        self.conv2 = nn.Conv2d(inter_plane, out_plane, 1, 1, 0)

        self.conv_dim = downsample
        self.stride = stride

        self.relu = nn.ReLU()

    def forward(self, x):

        identity = x

        out = self.relu(self.conv0(x))
        out = self.relu(self.conv1(out))
        out = self.conv2(out)

        if self.conv_dim is not None:
            identity = self.conv_dim(x)
        out += identity

        out = self.relu(out)

        return out
    
class SliceNet(nn.Module):
    def __init__(self):
        super(SliceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 320, kernel_size = (3,256), stride = 1, padding=(1,0))
        self.bn = nn.BatchNorm2d(320)
        self.relu = nn.leakyReLU()
   
        self.maxpool = nn.MaxPool2d((1,3),stride=2,padding=(0,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)   
    
    
class wiser(nn.Module):
    def __init__(self):
        super(wiser, self).__init__()

        # 1st conv before any network block
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        # blocks
        block = BasicBlock
        inplanes = [64, 256, 512, 1024]
        outplanes = [256, 512, 1024, 2048]

        num_blocks = [3,4,6,3]

        self.group0 = self._make_layer(block, inplanes[0], outplanes[0], num_blocks[0], stride = 1)
        self.group1 = self._make_layer(block, inplanes[1], outplanes[1], num_blocks[1], stride = 2)
        self.group2 = self._make_layer(block, inplanes[2], outplanes[2], num_blocks[2], stride = 2)
        self.group3 = self._make_layer(block, inplanes[3], outplanes[3], num_blocks[3], stride = 2)

        #slice channel
        self.slicenet = SliceNet()
        
        #two-channel fusion
        self.fusion = nn.Linear(2048 + 320, 2048)
        
        #misc
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)

        #network initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplane, outplane, num_block, stride):
        #check for bottleneck and stride cases
        downsample = None
        if stride != 1 or inplane != outplane:
            downsample = nn.Conv2d(inplane, outplane, kernel_size = 1, stride = stride, padding = 0)

        #compose layers
        layers = []
        layers.append(block(inplane, outplane, stride = stride, downsample = downsample))

        for _ in range(1, num_block):
            layers.append(block(outplane, outplane))

        return nn.Sequential(*layers)

    def forward(self, x):
        #compute for residual channel
        [out, a] = self.maxpool(self.relu(self.conv0(x)))
        out = self.group0(out)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        
        #compute for slice channel
        out2 = self.slicenet(x)
        #import ipdb; ipdb.set_trace()
        #fuse channel results
        out_fusion = self.fusion(torch.cat([out,out2],1))

        return out_fusion, a