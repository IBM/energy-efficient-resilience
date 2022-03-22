import torch
from torch import nn
import torch.nn.functional as F

# CIFAR10
# class LeNet(nn.Module):
#    def __init__(self):
#        super(LeNet, self).__init__()
#        self.conv1 = nn.Conv2d(3, 6, 5)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.fc1   = nn.Linear(16*5*5, 120)
#        self.fc2   = nn.Linear(120, 84)
#        self.fc3   = nn.Linear(84, 10)
#
#    def forward(self, x):
#        out = F.relu(self.conv1(x))
#        out = F.max_pool2d(out, 2)
#        out = F.relu(self.conv2(out))
#        out = F.max_pool2d(out, 2)
#        out = out.view(out.size(0), -1)
#        out = F.relu(self.fc1(out))
#        out = F.relu(self.fc2(out))
#        out = self.fc3(out)
#        return out
#
# MNIST


class LeNet(nn.Module):
    def __init__(self, input_channels, classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        output = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return output


def lenet(input_channels, classes):
    model = LeNet(input_channels, classes)
    return model
