import torch
import torch.nn as nn
import torch.nn.functional as F
from faultinjection_ops import zs_faultinjection_ops
from quantized_ops import zs_quantized_ops

conv_clamp_val = 0.05
fc_clamp_val = 0.1

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, precision, ber, position, faulty_layers):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        if "conv" in faulty_layers:
            self.conv1 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        self.bn2 = nn.BatchNorm2d(planes)
        if "conv" in faulty_layers:
            self.conv2 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv2 = zs_quantized_ops.nnConv2dSymQuant_op(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )

        if stride != 1 or in_planes != self.expansion*planes:
            if "conv" in faulty_layers:
                self.shortcut = nn.Sequential(
                    zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                        precision=precision,
                        clamp_val=conv_clamp_val,
                    ),
                )
            else:
                self.shortcut = nn.Sequential(
                    zs_quantized_ops.nnConv2dSymQuant_op(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                        precision=precision,
                        clamp_val=conv_clamp_val,
                    ),
                )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride, precision, ber, position, faulty_layers):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        if "conv" in faulty_layers:
            self.conv1 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                in_planes,
                planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_planes,
                planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        self.bn2 = nn.BatchNorm2d(planes)
        if "conv" in faulty_layers:
            self.conv2 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                planes,
                planes,
                kernel_size=3,
                stride=stride, 
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv2 = zs_quantized_ops.nnConv2dSymQuant_op(
                planes,
                planes,
                kernel_size=3,
                stride=stride, 
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        self.bn3 = nn.BatchNorm2d(planes)
        if "conv" in faulty_layers:
            self.conv3 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                planes,
                self.expansion*planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv3 = zs_quantized_ops.nnConv2dSymQuant_op(
                planes,
                self.expansion*planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )

        if stride != 1 or in_planes != self.expansion*planes:
            if "conv" in faulty_layers:
                self.shortcut = nn.Sequential(
                    zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                        precision=precision,
                        clamp_val=conv_clamp_val,
                    ),
                )
            else:
                self.shortcut = nn.Sequential(
                    zs_quantized_ops.nnConv2dSymQuant_op(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                        precision=precision,
                        clamp_val=conv_clamp_val,
                    ),
                )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, precision, ber, position, faulty_layers):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        if "conv" in faulty_layers:
            self.conv1 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv1 = zs_quantized_ops.nnConv2dSymQuant_op(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, precision=precision, ber=ber, position=position, faulty_layers=faulty_layers)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, precision=precision, ber=ber, position=position, faulty_layers=faulty_layers)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, precision=precision, ber=ber, position=position, faulty_layers=faulty_layers)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, precision=precision, ber=ber, position=position, faulty_layers=faulty_layers)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        if "linear" in faulty_layers:
            self.linear = zs_quantized_ops.nnLinearSymQuant_op(512 * block.expansion, num_classes, precision, fc_clamp_val)
        else:
            self.linear = zs_quantized_ops.nnLinearSymQuant_op(
                512 * block.expansion, num_classes, precision, fc_clamp_val
            )

    def _make_layer(self, block, planes, num_blocks, stride, precision, ber, position, faulty_layers):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, precision, ber, position, faulty_layers))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(classes, precision, ber, position, faulty_layers):
    return PreActResNet(PreActBlock, [2,2,2,2], classes, precision, ber, position, faulty_layers)

def PreActResNet34(classes, precision, ber, position, faulty_layers):
    return PreActResNet(PreActBlock, [3,4,6,3], precision, ber, position, faulty_layers)

def PreActResNet50(classes, precision, ber, position, faulty_layers):
    return PreActResNet(PreActBottleneck, [3,4,6,3], precision, ber, position, faulty_layers)

def PreActResNet101(classes, precision, ber, position, faulty_layers):
    return PreActResNet(PreActBottleneck, [3,4,23,3], precision, ber, position, faulty_layers)

def PreActResNet152(classes, precision, ber, position, faulty_layers):
    return PreActResNet(PreActBottleneck, [3,8,36,3], precision, ber, position, faulty_layers)


def preactresnetf(
    arch,
    classes,
    precision,
    ber,
    position,
    faulty_layers,
):
    if arch == "preactresnet18":
        return PreActResNet18(
            classes,
            precision,
            ber,
            position,
            faulty_layers,
        )