"""
Code source: torchvision repository resnet code
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

RESNET_101 = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
RESNET_50 = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
RESNET_34 = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
RESNET_18 = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            dilation = 1

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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

    def __init__(self, block, layers, output_stride, norm_layer=nn.BatchNorm2d, input_channels=3):
        self.inplanes = 64

        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], norm_layer=norm_layer)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], norm_layer=norm_layer)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], nn.BatchNorm2d=nn.BatchNorm2d)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation, downsample=downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion

        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, dilation=blocks[i]*dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_feat


def ResNet18(output_stride, norm_layer=nn.BatchNorm2d, pretrained=True, input_channels=3):
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, norm_layer=norm_layer, input_channels=input_channels)

    if pretrained:
        _load_pretrained_model(model, RESNET_18)

    return model


def ResNet34(output_stride, norm_layer=nn.BatchNorm2d, pretrained=True, input_channels=3):
    model = ResNet(BasicBlock, [3, 4, 23, 3], output_stride, norm_layer=norm_layer, input_channels=input_channels)

    if pretrained:
        _load_pretrained_model(model, RESNET_34)

    return model

def ResNet101(output_stride, norm_layer=nn.BatchNorm2d, pretrained=True, input_channels=3):
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, norm_layer=norm_layer, input_channels=input_channels)

    if pretrained:
        _load_pretrained_model(model, RESNET_101)

    return model


def ResNet50(output_stride, norm_layer=nn.BatchNorm2d, pretrained=True, input_channels=3):
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, norm_layer=norm_layer, input_channels=input_channels)

    if pretrained:
        _load_pretrained_model(model, RESNET_50)

    return model


def _load_pretrained_model(model, url):
    pretrain_dict = model_zoo.load_url(url)
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)