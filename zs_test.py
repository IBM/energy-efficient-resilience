
import numpy as np
import os 
import sys
import argparse
import pdb
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import zs_hooks_stats as stats
sys.path.append('./models')

debug=False
visualize = False

faultinject = True

def inference(testloader, model, arch, dataset, voltage, ber, precision, position, error_type, checkpoint_path, faulty_layers, iters, device):

    if arch == 'resnet18' or arch == 'resnet34':
        stats.resnet_register_hooks(model,arch)
    if arch == 'vgg16' or arch == 'vgg11':
        stats.vgg_register_hooks(model,arch)
    logger = stats.DataLogger(int(len(testloader.dataset)/testloader.batch_size), testloader.batch_size)

    

    print(model)
    model = model.to(device)
    #model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    print('Restoring model from checkpoint', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    print('restored checkpoint at epoch - ',checkpoint['epoch'])
    #print('Training loss =', checkpoint['loss'])
    print('Training accuracy =', checkpoint['accuracy'])

    model.eval()

    model=model.to(device)
    running_correct = 0.0

    with torch.no_grad():
        for t, (inputs,classes) in enumerate(testloader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            model_outputs =model(inputs)
            lg, preds = torch.max(model_outputs, 1)
            correct=torch.sum(preds == classes.data)
            running_correct += correct
            
            logger.update(model_outputs)

    print('Eval Accuracy %.3f'%(running_correct.double()/(len(testloader.dataset))))

    #if arch=='resnet18' or arch=='resnet34':
    #    stats.inspect_model(model)
    #    stats.resnet_print_stats()
    #elif arch=='vgg16':
    #    stats.inspect_model(model)
    #    stats.vgg_print_stats()

    #logger.visualize()
