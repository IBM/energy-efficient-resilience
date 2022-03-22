# CIFAR10
import torch
import torch.nn.functional as F
from torch import nn

# import common.torch


class SimpleNet(nn.Module):
    def __init__(self, input_channels=3, classes=10, bn=False, dropout=False):
        super(SimpleNet, self).__init__()
        self.bn = bn
        self.dropout = dropout
        self.in_channels = input_channels
        self.features = self._make_layers()
        self.classifier = nn.Linear(256, classes)
        # self.drp = nn.Dropout(0.1)

    # def load_my_state_dict(self, state_dict):

    #    own_state = self.state_dict()

    #    # print(own_state.keys())
    #    # for name, val in own_state:
    #    # print(name)
    #    for name, param in state_dict.items():
    #        name = name.replace('module.', '')
    #        if name not in own_state:
    #            # print(name)
    #            continue
    #        if isinstance(param, Parameter):
    #            # backwards compatibility for serialized parameters
    #            param = param.data
    #        print("STATE_DICT: {}".format(name))
    #        try:
    #            own_state[name].copy_(param)
    #        except:
    #            print('While copying the parameter named {},"
    #            " whose dimensions in the model are'
    #                  ' {} and whose dimensions in the "
    #                  "checkpoint are {}, ... Using Initial Params'.format(
    #                name, own_state[name].size(), param.size()))

    def forward(self, x):
        out = self.features(x)
        # Global Max Pooling
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        # out = F.dropout2d(out, 0.1, training=True)
        # out = self.drp(out)
        # out = out.view(out.size(0), -1)

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers = []
        layers += [
            nn.Conv2d(
                self.in_channels,
                64,
                kernel_size=[3, 3],
                stride=(1, 1),
                padding=(1, 1),
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.Conv2d(
                64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.Conv2d(
                128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.Conv2d(
                128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                dilation=(1, 1),
                ceil_mode=False,
            )
        ]
        if self.dropout:
            layers += [nn.Dropout2d(p=0.1)]

        layers += [
            nn.Conv2d(
                128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.Conv2d(
                128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.Conv2d(
                128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                dilation=(1, 1),
                ceil_mode=False,
            )
        ]
        if self.dropout:
            layers += [nn.Dropout2d(p=0.1)]

        layers += [
            nn.Conv2d(
                256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.Conv2d(
                256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                dilation=(1, 1),
                ceil_mode=False,
            )
        ]
        if self.dropout:
            layers += [nn.Dropout2d(p=0.1)]

        layers += [
            nn.Conv2d(
                256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                dilation=(1, 1),
                ceil_mode=False,
            )
        ]
        if self.dropout:
            layers += [nn.Dropout2d(p=0.1)]

        layers += [
            nn.Conv2d(
                512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.Conv2d(
                2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        layers += [
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                dilation=(1, 1),
                ceil_mode=False,
            )
        ]
        if self.dropout:
            layers += [nn.Dropout2d(p=0.1)]

        layers += [
            nn.Conv2d(
                256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
            )
        ]
        if self.bn:
            layers += [
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
            ]
        layers += [nn.ReLU(inplace=True)]

        # layers += [nn.MaxPool2d(kernel_size=[32, 32))]

        model = nn.Sequential(*layers)

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(
                # m.weight.data, gain=nn.init.calculate_gain('relu'))
                nn.init.kaiming_uniform_(
                    m.weight.data, nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return model


def simplenet(input_channels, classes):
    model = SimpleNet(input_channels=input_channels, classes=classes)
    return model


def test():
    net = simplenet(1, 10)
    params = list(net.parameters())
    print(len(params))
    print(net)
