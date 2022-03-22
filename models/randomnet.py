import torch
from torch import nn

# sys.path.append('../quantized_ops')
# from quantized_ops import zs_quantized_ops
# conv_clamp_val=0.1
# fc_clamp_val=0.1


class RandomNet(nn.Module):
    def __init__(self, input_channels, classes, precision):
        super(RandomNet, self).__init__()
        #        if (precision < 0):
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        #        else:
        #            self.conv1 = zs_quantized_ops.nnConv2dSymQuant_op(
        #            input_channels, 32, kernel_size=3, stride=1,
        #            padding=0, bias=True, precision=precision,
        #            clamp_val=conv_clamp_val)
        #        if (precision < 0):
        self.fc1 = nn.Linear(28800, 128)
        self.fc2 = nn.Linear(128, classes)
        #        else:
        #            self.fc1 = zs_quantized_ops.nnLinearSymQuant_op(
        #            28800,128,precision, fc_clamp_val)
        #            self.fc2 = zs_quantized_ops.nnLinearSymQuant_op(
        #            128,classes,precision, conv_clamp_val)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        #    x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu2(x)
        output = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return output


def randomnet(input_channels, classes, precision):
    model = RandomNet(input_channels, classes, precision)
    return model


def test():
    net = randomnet(3, 10, -1)
    print(net)
    for name, layer in net.named_modules():
        print(name, layer)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
