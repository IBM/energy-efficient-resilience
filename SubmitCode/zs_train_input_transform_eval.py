import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict

from config import cfg
from models import init_models_pairs, create_faults, init_models
from models.generator import *
import faultsMap as fmap

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import itertools
import numpy as np
import tqdm
import copy


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-20
PGD_STEP = 2


# Get the activation from clean model.
def get_activation_c(act_c, name):
    def hook(model, input, output):
        act_c[name] = output
    return hook

# Get the activation from perturbed model.
def get_activation_p(act_p, name):
    def hook(model, input, output):
        act_p[name] = output
    return hook

def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(
        labels.size(0)
    )  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds

def accuracy_checking_clean(model_orig, trainloader, testloader, device):
    """
      Calculating the clean accuracy.
      :param model_orig: Clean model.
      :param trainloader: The loader of training data.
      :param testloader: The loader of testing data.
      :param device: Specify GPU usage.
    """
    cfg.replaceWeight = False
    total_train = 0
    total_test = 0
    correct_orig_train = 0
    correct_orig_test = 0

    # For training data:
    for x, y in trainloader:
        total_train += 1
        x, y = x.to(device), y.to(device)
        out_orig = model_orig(x)
        _, pred_orig = out_orig.max(1)
        y = y.view(y.size(0))
        correct_orig_train += torch.sum(pred_orig == y.data).item()
    accuracy_orig_train = correct_orig_train / (len(trainloader.dataset))

    # For testing data:
    for x, y in testloader:
        total_test += 1
        x, y = x.to(device), y.to(device)
        out_orig = model_orig(x)
        _, pred_orig = out_orig.max(1)
        y = y.view(y.size(0))
        correct_orig_test += torch.sum(pred_orig == y.data).item()
    accuracy_orig_test = correct_orig_test / (len(testloader.dataset))

    print("Accuracy of training data: clean model: {:5f}".format(accuracy_orig_train))
    print("Accuracy of testing data: clean model: {:5f}".format(accuracy_orig_test))


def accuracy_checking(model_orig, model_p, trainloader, testloader, transform_model, device, use_transform=False):
    """
      Calculating the accuracy with given clean model and perturbed model.
      :param model_orig: Clean model.
      :param model_p: Perturbed model.
      :param trainloader: The loader of training data.
      :param testloader: The loader of testing data.
      :param transform_model: The object of transformation model.
      :param device: Specify GPU usage.
      :use_transform: Should apply input transformation or not.
    """
    cfg.replaceWeight = False
    total_train = 0
    total_test = 0
    correct_orig_train = 0
    correct_p_train = 0
    correct_orig_test = 0
    correct_p_test = 0

    # For training data:
    for x, y in trainloader:
        total_train += 1
        x, y = x.to(device), y.to(device)
        if use_transform:
            x_adv = transform_model(x)
            out_orig = model_orig(x_adv)
            out_p = model_p(x_adv)
        else:
            out_orig = model_orig(x)
            out_p = model_p(x)
        _, pred_orig = out_orig.max(1)
        _, pred_p = out_p.max(1)
        y = y.view(y.size(0))
        correct_orig_train += torch.sum(pred_orig == y.data).item()
        correct_p_train += torch.sum(pred_p == y.data).item()
    accuracy_orig_train = correct_orig_train / (len(trainloader.dataset))
    accuracy_p_train = correct_p_train / (len(trainloader.dataset))

    # For testing data:
    for x, y in testloader:
        total_test += 1
        x, y = x.to(device), y.to(device)
        if use_transform:
            x_adv = transform_model(x)
            out_orig = model_orig(x_adv)
            out_p = model_p(x_adv)
        else:
            out_orig = model_orig(x)
            out_p = model_p(x)
        _, pred_orig = out_orig.max(1)
        _, pred_p = out_p.max(1)
        y = y.view(y.size(0))
        correct_orig_test += torch.sum(pred_orig == y.data).item()
        correct_p_test += torch.sum(pred_p == y.data).item()
    accuracy_orig_test = correct_orig_test / (len(testloader.dataset))
    accuracy_p_test = correct_p_test / (len(testloader.dataset))

    if use_transform:
        print('----- With input transformation: -----')
    else:
        print('----- Without using input transformation: -----')

    print("Accuracy of training data: clean model:{:5f}, perturbed model: {:5f}".format(accuracy_orig_train, accuracy_p_train))
    print("Accuracy of testing data: clean model:{:5f}, perturbed model: {:5f}".format(accuracy_orig_test, accuracy_p_test))

    return accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test

def transform_eval(trainloader, testloader, arch, dataset, in_channels, precision, checkpoint_path, force, device, fl, ber, pos,):
    """
        Apply quantization aware training.
        :param trainloader: The loader of training data.
        :param in_channels: An int. The input channels of the training data.
        :param arch: A string. The architecture of the model would be used.
        :param dataset: A string. The name of the training data.
        :param ber: A float. How many rate of bits would be attacked.
        :param precision: An int. The number of bits would be used to quantize
                          the model.
        :param position:
        :param checkpoint_path: A string. The path that stores the models.
        :param device: Specify GPU usage.
    """
    
    torch.backends.cudnn.benchmark = True

    if(cfg.testing_mode == 'clean'):
        model, checkpoint_epoch = init_models(arch, 3, precision, True, checkpoint_path, dataset)

        model = model.to(device)
        model.eval()
        accuracy_checking_clean(model, trainloader, testloader, device)
    
    if cfg.testing_mode == 'generator_base':
        
        if cfg.G == 'ConvL':
            Gen = GeneratorConvLQ(precision)
        elif cfg.G == 'ConvS':
            Gen = GeneratorConvSQ(precision)
        elif cfg.G == 'DeConvL':
            Gen = GeneratorDeConvLQ(precision)
        elif cfg.G == 'DeConvS':
            Gen = GeneratorDeConvSQ(precision)
        elif cfg.G == 'UNetL':
            Gen = GeneratorUNetLQ(precision)
        elif cfg.G == 'UNetS':
            Gen = GeneratorUNetSQ(precision)
        
        Gen.load_state_dict(torch.load(cfg.G_PATH))
        # Gen = torch.load(cfg.G_PATH)
        Gen = Gen.to(device)
        print('Successfully loading the generator model.')

        print('========== Start checking the accuracy with different perturbed model: bit error mode ==========')
        # Setting without input transformation
        accuracy_orig_train_list = []
        accuracy_p_train_list = []
        accuracy_orig_test_list = []
        accuracy_p_test_list = []
    
        # Setting with input transformation
        accuracy_orig_train_list_with_transformation = []
        accuracy_p_train_list_with_transformation = []
        accuracy_orig_test_list_with_transformation = []
        accuracy_p_test_list_with_transformation = []
    
        for i in range(50000, 50010):
            print(' ********** For seed: {} ********** '.format(i))
            (model, checkpoint_epoch, model_perturbed, checkpoint_epoch_perturbed) = init_models_pairs( 
                        arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=i, dataset=dataset)
            model, model_perturbed = model.to(device), model_perturbed.to(device),
            fmap.BitErrorMap0to1 = None 
            fmap.BitErrorMap1to0 = None
            create_faults(precision, ber, pos, seed=i)
            model.eval()
            model_perturbed.eval()
            Gen.eval()
    
            # Without using transform
            accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Gen, device, use_transform=False)
            accuracy_orig_train_list.append(accuracy_orig_train)
            accuracy_p_train_list.append(accuracy_p_train)
            accuracy_orig_test_list.append(accuracy_orig_test)
            accuracy_p_test_list.append(accuracy_p_test)
    
            # With input transform
            accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Gen, device, use_transform=True)
            accuracy_orig_train_list_with_transformation.append(accuracy_orig_train)
            accuracy_p_train_list_with_transformation.append(accuracy_p_train)
            accuracy_orig_test_list_with_transformation.append(accuracy_orig_test)
            accuracy_p_test_list_with_transformation.append(accuracy_p_test)


        # Without using transform
        print('The average results without input transformation -> accuracy_orig_train: {:5f}, accuracy_p_train: {:5f}, accuracy_orig_test: {:5f}, accuracy_p_test: {:5f}'.format(
          np.mean(accuracy_orig_train_list), np.mean(accuracy_p_train_list), np.mean(accuracy_orig_test_list), np.mean(accuracy_p_test_list)
        ))
        print('The average results without input transformation -> std_accuracy_orig_train: {:5f}, std_accuracy_p_train: {:5f}, std_accuracy_orig_test: {:5f}, std_accuracy_p_test: {:5f}'.format(
          np.std(accuracy_orig_train_list), np.std(accuracy_p_train_list), np.std(accuracy_orig_test_list), np.std(accuracy_p_test_list)
        ))
    
        print()
    
        # With input transform
        print('The average results with input transformation -> accuracy_orig_train: {:5f}, accuracy_p_train: {:5f}, accuracy_orig_test: {:5f}, accuracy_p_test: {:5f}'.format(
          np.mean(accuracy_orig_train_list_with_transformation), np.mean(accuracy_p_train_list_with_transformation), np.mean(accuracy_orig_test_list_with_transformation), np.mean(accuracy_p_test_list_with_transformation)
        ))
        print('The average results with input transformation -> std_accuracy_orig_train: {:5f}, std_accuracy_p_train: {:5f}, std_accuracy_orig_test: {:5f}, std_accuracy_p_test: {:5f}'.format(
          np.std(accuracy_orig_train_list_with_transformation), np.std(accuracy_p_train_list_with_transformation), np.std(accuracy_orig_test_list_with_transformation), np.std(accuracy_p_test_list_with_transformation)
        ))

    if cfg.testing_mode == 'visualization':
        
        # Set plotting config
        classNums = 0
        if dataset == 'cifar10':
            classNums = 10
        elif dataset == 'cifar100':
            classNums = 100
        elif dataset == 'tinyimagenet':
            classNums = 200
        elif dataset == 'gtsrb':
            classNums = 43
        else:
            classNums = 10

        if dataset == 'cifar10':
            cmap = plt.cm.get_cmap('tab10')
        else:
            cmap = plt.cm.get_cmap('nipy_spectral')
        colorList = [cmap(1/classNums*i) for i in range(classNums)] # Choose the colors

        def tsneNormalization(x):
            value_range = (np.max(x) - np.min(x))
            starts_from_zero = x - np.min(x)
            return starts_from_zero / value_range

        if cfg.G == 'ConvL':
            Gen = GeneratorConvLQ(precision)
        elif cfg.G == 'ConvS':
            Gen = GeneratorConvSQ(precision)
        elif cfg.G == 'DeConvL':
            Gen = GeneratorDeConvLQ(precision)
        elif cfg.G == 'DeConvS':
            Gen = GeneratorDeConvSQ(precision)
        elif cfg.G == 'UNetL':
            Gen = GeneratorUNetLQ(precision)
        elif cfg.G == 'UNetS':
            Gen = GeneratorUNetSQ(precision)
        
        Gen.load_state_dict(torch.load(cfg.G_PATH))
        Gen = Gen.to(device)
        
        (model, checkpoint_epoch, model_perturbed, checkpoint_epoch_perturbed) = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=cfg.tsneModel, dataset=dataset)
        model, model_perturbed = model.to(device), model_perturbed.to(device),
        fmap.BitErrorMap0to1 = None 
        fmap.BitErrorMap1to0 = None
        create_faults(precision, ber, pos, seed=cfg.tsneModel)
        model.eval()
        model_perturbed.eval()
        Gen.eval()

        accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Gen, device, use_transform=False)
        accuracy_orig_trainG, accuracy_p_trainG, accuracy_orig_testG, accuracy_p_testG = accuracy_checking(model, model_perturbed, trainloader, testloader, Gen, device, use_transform=True)

        activation_c, activation_p = {}, {}
        if 'vgg' in arch:
            for name, layer in model.named_modules():
                if 'classifier' in name:
                    layer.register_forward_hook(get_activation_c(activation_c, name))
    
            for name, layer in model_perturbed.named_modules():
                if 'classifier' in name:
                    layer.register_forward_hook(get_activation_p(activation_p, name))
        elif 'resnet' in arch: 
            for name, layer in model.named_modules():
                if 'linear' in name:
                    layer.register_forward_hook(get_activation_c(activation_c, name))
            for name, layer in model_perturbed.named_modules():
                if 'linear' in name:
                    layer.register_forward_hook(get_activation_p(activation_p, name))

        cleanOutput, cleanOutputG, perturbedOutput, perturbedOutputG, labelList = None, None, None, None, None
        
        print('========== Plotting TSNE results for clean model and perturbed model without using generator ==========')
        for batch_id, (image, label) in tqdm.tqdm(enumerate(testloader)):
            image, label = image.to(device), label.to(device)
            _, _ = model(image), model_perturbed(image)

            layer_keys = activation_c.keys()
            for name in layer_keys:
                if cleanOutput == None and perturbedOutput == None:
                    cleanOutput = activation_c[name].data
                    perturbedOutput = activation_p[name].data
                    labelList = label
                else:
                    cleanOutput = torch.concat((cleanOutput, activation_c[name].data))
                    perturbedOutput = torch.concat((perturbedOutput, activation_p[name].data))
                    labelList = torch.concat((labelList, label))

        #  Plot results
        tsneC = TSNE(n_components=2, random_state=0).fit_transform(cleanOutput.cpu())
        tsnexC = tsneNormalization(tsneC[:, 0])
        tsneyC = tsneNormalization(tsneC[:, 1])
        plt.figure(figsize=(15,15))
        for idx in range(classNums):
            indices = [i for i, l in enumerate(labelList) if idx == l]
            current_tx = np.take(tsnexC, indices)
            current_ty = np.take(tsneyC, indices)
            plt.scatter(current_tx, current_ty, color=colorList[idx], label=str(idx+1))
        plt.legend(loc='upper left', fontsize=20)
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.title('Clean model without NeuralFuse. Accuracy={}%'.format(np.round(accuracy_orig_test*100, decimals=2)) , fontsize=25)
        plt.savefig('./tsne/{}_{}_tsneC.jpg'.format(arch, dataset))
        plt.clf()


        tsneP = TSNE(n_components=2, random_state=0).fit_transform(perturbedOutput.cpu())
        tsnexP = tsneNormalization(tsneP[:, 0])
        tsneyP = tsneNormalization(tsneP[:, 1])
        plt.figure(figsize=(15,15))
        for idx in range(classNums):
            indices = [i for i, l in enumerate(labelList) if idx == l]
            current_tx = np.take(tsnexP, indices)
            current_ty = np.take(tsneyP, indices)
            plt.scatter(current_tx, current_ty, color=colorList[idx], label=str(idx+1))
        plt.legend(loc='upper left', fontsize=20)
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.title('Perturbed model without NeuralFuse. Accuracy={}%'.format(np.round(accuracy_p_test*100, decimals=2)), fontsize=25)
        plt.savefig('./tsne/{}_{}_ber{}_tsneP.jpg'.format(arch, dataset, ber))
        plt.clf()
        
        print('========== Plotting TSNE results for clean model and perturbed model with using generator ==========')
    
        fileName = cfg.G_PATH.split('/')[-1][:-3]
        
        for batch_id, (image, label) in tqdm.tqdm(enumerate(testloader)):
            image, label = image.to(device), label.to(device)
            image_adv = Gen(image)
            image_adv = image_adv.to(device)
            _, _ = model(image_adv), model_perturbed(image_adv)

            layer_keys = activation_c.keys()
            for name in layer_keys:
                if cleanOutputG == None and perturbedOutputG == None:
                    cleanOutputG = activation_c[name].data
                    perturbedOutputG = activation_p[name].data
                else:
                    cleanOutputG = torch.concat((cleanOutputG, activation_c[name].data))
                    perturbedOutputG = torch.concat((perturbedOutputG, activation_p[name].data))

        #  Plot results
        tsneCG = TSNE(n_components=2, random_state=0).fit_transform(cleanOutputG.cpu())
        tsnexCG = tsneNormalization(tsneCG[:, 0])
        tsneyCG = tsneNormalization(tsneCG[:, 1])
        plt.figure(figsize=(15,15))
        for idx in range(classNums):
            indices = [i for i, l in enumerate(labelList) if idx == l]
            current_tx = np.take(tsnexCG, indices)
            current_ty = np.take(tsneyCG, indices)
            plt.scatter(current_tx, current_ty, color=colorList[idx], label=str(idx+1))
        plt.legend(loc='upper left', fontsize=20)
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.title('Clean model with NeuralFuse. Accuracy={}%'.format(np.round(accuracy_orig_testG*100, decimals=2)), fontsize=25)
        plt.savefig('./tsne/{}_tsneCG.jpg'.format(fileName))
        plt.clf()

        tsnePG = TSNE(n_components=2, random_state=0).fit_transform(perturbedOutputG.cpu())
        tsnexPG = tsneNormalization(tsnePG[:, 0])
        tsneyPG = tsneNormalization(tsnePG[:, 1])
        plt.figure(figsize=(15,15))
        for idx in range(classNums):
            indices = [i for i, l in enumerate(labelList) if idx == l]
            current_tx = np.take(tsnexPG, indices)
            current_ty = np.take(tsneyPG, indices)
            plt.scatter(current_tx, current_ty, color=colorList[idx], label=str(idx+1))
        plt.legend(loc='upper left', fontsize=20)
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.title('Perturbed model with NeuralFuse. Accuracy={}%'.format(np.round(accuracy_p_testG*100, decimals=2)), fontsize=25)
        plt.savefig('./tsne/{}_tsnePG.jpg'.format(fileName))
        plt.clf()