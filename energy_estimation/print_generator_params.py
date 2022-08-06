import torch
from torch import nn
import re
import numpy as np
import sys
import csv
sys.path.append("../")
from models import generator 
from models import vggf
from models import resnetf
from config import cfg

import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

layer_counter=0
input_w = {}
input_h = {} 
input_c = {} 
filter_h = {} 
filter_w = {}
filter_c = {}
layer_name = {}

def collect_activation_sizes(self, input, output):
    global layer_counter, input_w, input_h, input_c, filter_h, filter_w, filter_c
    if 'Conv2d' in self.__class__.__name__:
        i_shape = list(input[0].shape)
        f_shape = list(self.weight.shape)
        input_w[layer_counter] = i_shape[3]
        input_h[layer_counter] = i_shape[2]
        input_c[layer_counter] = i_shape[1]
        filter_w[layer_counter] = f_shape[3]
        filter_h[layer_counter] = f_shape[2]
        filter_c[layer_counter] = f_shape[0]
        layer_name[layer_counter] = self.module_name
        layer_counter +=1

    
    
def main():
    dataset = "cifar10"
    in_channels = 3
    classes = 10
    #model = generator.GeneratorUNetSQ(precision = 8)
    #model = resnetf("resnet18", classes, 8, 0, 0, [])
    model = vggf("D", in_channels, classes, True, 8, 0, 0, [])
    print(model)
    model.eval()
    model.to(device)
    hooks = {}
    for name, module in model.named_modules():
        module.module_name = name
        hooks[name] = module.register_forward_hook(collect_activation_sizes)


    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #transforms.Lambda(lambda t: t * 2 - 1),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=transform_test,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=2,
    )

    for batch_id, (image, label) in enumerate(testloader):
        if (batch_id == 0):
            image, label = image.to(device), label.to(device)
            y = model(image)

    csvfile = 'output.csv'
    file = open(csvfile, 'a+')
    with file:
        header = ['Layer name', 'IFMAP Height','IFMAP Width', 'Filter Height', 'Filter Width', 'Channels', 'Num Filter', 'Strides']
        write = csv.writer(file)
        write.writerow(header)
        for l in range(layer_counter):
            row = []
            row.append(layer_name[l])
            row.append(input_h[l])
            row.append(input_w[l])
            row.append(filter_h[l])
            row.append(filter_w[l])
            row.append(input_c[l])
            row.append(filter_c[l])
            row.append('1')
            write.writerow(row)

    
    print(input_w)
    print(input_h)
    print(input_c)
    print(filter_w)
    print(filter_h)
    print(filter_c)
    print(layer_name)

if __name__ == "__main__":
    main()
