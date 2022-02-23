import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import matplotlib.pyplot as plt

import sys, os, argparse
import pdb
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import zs_train as train
import zs_test as test
import zs_weight_prune as wprune

import torchvision
import torchvision.transforms as transforms

sys.path.append('./faultmodels')
from faultmodels import randomfault

sys.path.append('./models')
from models import vgg
from models import resnet
from models import lenet
from models import lenetf
from models import vggf
from models import resnetf
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs=60
batch_size=128
test_batch_size=100

datafolder = '../ZStressmark/data'

def main(argv):

    print('Running command:', str(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("arch", help="<resnet18> <resnet34> <vgg11> <vgg16> <lenet> specify network architecture", default='resnet18')  
    parser.add_argument("mode", help="<train> <prune> <eval> specify operation", default='eval')  
    parser.add_argument("dataset", help="<fashion> <cifar10> <mnist> specify dataset", default='fashion')  
    parser.add_argument("-p", "--precision", type=int, help="bit precision <8,16,32,-1> -1 is float32", default=8)
    parser.add_argument("-v", "--voltage", type=int, help="memory voltage levels V1<V2<V3<V4", default=4)
    parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate for training", default=0.01)
    parser.add_argument("-ber", "--bit_error_rate", type=float, help="bit error rate for training corresponding to known voltage", default=0.01)
    parser.add_argument("-pos", "--position", type=int, help="specify position of bit errors", default=-1)
    parser.add_argument("-err", "--error_type", type=int, help="Inject hardware errors in weights and inputs <0 -- No Error, 1-- Fixed faultmap injection into weights, 2--Fixed faultmap injection into inputs 3-- Both, 4-Random faultmap injection in weights, 5--Random faultmap injection in inputs, 6-- Both>", default=0)
    parser.add_argument("-rt", "--retrain", action="store_true", help="retrain on top of already trained model", default = False)
    parser.add_argument("-cp", "--checkpoint", help="Name of the stored model that needs to be retrained", default='./resnet18_checkpoints/resnet_resnet_model_119.pth') 
    parser.add_argument("-iters", "--eval_iters", type=int, help="Number of eval iterations", default=1)
    args = parser.parse_args()


    #if args.position>args.precision-1:
    #    print('ERROR: specified bit position for error exceeds the precision')
    #    exit(0)

    print('Preparing data..', args.dataset)
    if (args.dataset == 'cifar10'):
        dataset = 'cifar'    
        in_channels=3
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        trainset = torchvision.datasets.CIFAR10(root=datafolder, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=datafolder, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)


    elif (args.dataset == 'mnist'):
        dataset = 'mnist'
        in_channels=1
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

        trainset = torchvision.datasets.MNIST(root=datafolder, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root=datafolder, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    else:
        dataset = 'fashion'
        in_channels=1
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))   # per channel means and std devs
    ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2868,), (0.3524,))
    ])

        trainset = torchvision.datasets.FashionMNIST(root=datafolder, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.FashionMNIST(root=datafolder, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
            

    print('Device', device)

    if args.arch == 'vgg11':
        model  = vgg('A',in_channels, 10, True, args.precision)
    elif args.arch == 'vgg16':
        model  = vgg('D',in_channels, 10, True, args.precision)
    elif args.arch == 'resnet18':
        model = resnet('resnet18', 10, args.precision) 
    elif args.arch == 'resnet34':
        model = resnet('resnet34', 10, args.precision) 
    else:
        model = lenet(in_channels,10,args.precision)

    faulty_layers = []
    if (args.error_type > 0):
        prec = 8 # Currently fault injection supported only for 8 bit quantization
        rf = randomfault.RandomFaultModel(args.bit_error_rate, prec, args.position, 0)
        BitErrorMap0 = torch.tensor(rf.BitErrorMap_flip0).to(torch.int32).to(device)
        BitErrorMap1 = torch.tensor(rf.BitErrorMap_flip1).to(torch.int32).to(device)
        #faulty_layers = ['conv', 'linear']
        faulty_layers = ['linear']
        if args.arch == 'vgg11':
            model  = vggf('A',in_channels, 10, True, prec, args.bit_error_rate, args.position, BitErrorMap0, BitErrorMap1, faulty_layers)
        elif args.arch == 'vgg16':
            model  = vggf('D',in_channels, 10, True, prec, args.bit_error_rate, args.position, BitErrorMap0, BitErrorMap1, faulty_layers)
        elif args.arch == 'resnet18':
            model = resnetf('resnet18', 10, prec, args.bit_error_rate, args.position, BitErrorMap0, BitErrorMap1, faulty_layers) 
        elif args.arch == 'resnet34':
            model = resnetf('resnet34', 10, prec, args.bit_error_rate, args.position, BitErrorMap0, BitErrorMap1, faulty_layers) 
        else:
            model = lenetf(in_channels,10,prec, args.bit_error_rate, args.position, BitErrorMap0, BitErrorMap1, faulty_layers)

    if (args.mode == 'train'):
        print('training args', args)
        train.training(trainloader, model, args.arch, dataset, args.learning_rate, epochs, args.voltage, args.bit_error_rate, args.precision, args.position, args.error_type, args.retrain, args.checkpoint, faulty_layers, device)
    elif (args.mode == 'prune'):
        wprune.pruneweights(trainloader, model, args.arch, dataset, args.precision, args.checkpoint, device)
    else:
        test.inference(testloader, model, args.arch, dataset, args.voltage, args.bit_error_rate, args.precision, args.position, args.error_type, args.checkpoint, faulty_layers, args.eval_iters, device)

if __name__ == "__main__":
    main(sys.argv[1:])
