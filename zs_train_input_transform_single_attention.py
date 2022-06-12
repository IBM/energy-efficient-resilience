# Copyright 2022 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from config import cfg
from models import init_models_pairs
import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy
import tqdm

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


class Self_Attention(nn.Module):
    def __init__(self):
        super(Self_Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.key = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 1)
        self.query = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 1)
        self.value = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 1)
        self.self_att = nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = 1)
        # self.gamma    = nn.Parameter(torch.tensor(0.0))
        self.softmax  = nn.Softmax(dim=1)

    def forward(self, image):
        """
            inputs:
                x: input feature map [Batch, Channel, Height, Width]
            returns:
                out: self attention value + input feature
                attention: [Batch, Channel, Height, Width]
        """
        x = image.data.clone()
        # x = self.relu1(self.bn1(self.conv1(x)))
        # x = self.relu2(self.bn2(self.conv2(x)))

        batchsize, C, H, W = x.size()
        N = H * W                                       # Number of features
        f_x = self.key(x).view(batchsize, -1, N)      # Keys                  [B, C_bar, N]
        g_x = self.query(x).view(batchsize, -1, N)      # Queries               [B, C_bar, N]
        h_x = self.value(x).view(batchsize, -1, N)      # Values                [B, C_bar, N]

        s =  torch.bmm(f_x.permute(0,2,1), g_x)         # Scores                [B, N, N]
        beta = self.softmax(s)                          # Attention Map         [B, N, N]

        v = torch.bmm(h_x, beta)                        # Value x Softmax       [B, C_bar, N]
        v = v.view(batchsize, -1, H, W)                 # Recover input shape   [B, C_bar, H, W]
        o = self.self_att(v)                            # Self-Attention output [B, C, H, W]
        
        # x_adv = o + image
        return torch.clamp(o, min=-1, max=1)


def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(
        labels.size(0)
    )  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds


def accuracy_checking(
    model_orig, model_p, trainloader, testloader, attention, device, use_transform=False
):
    # For training data first:
    total_train = 0
    total_test = 0
    correct_orig_train = 0
    correct_p_train = 0
    correct_orig_test = 0
    correct_p_test = 0
    for x, y in trainloader:
        total_train += 1
        x, y = x.to(device), y.to(device)
        if use_transform:
            x_adv = attention(x)
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

    for x, y in testloader:
        total_test += 1
        x, y = x.to(device), y.to(device)
        if use_transform:
            x_adv = attention(x)
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

    print(
        "Accuracy of training data: clean model:"
        "{:5f}, perturbed model: {:5f}".format(
            accuracy_orig_train, accuracy_p_train
        )
    )
    print(
        "Accuracy of testing data: clean model:"
        "{:5f}, perturbed model: {:5f}".format(
            accuracy_orig_test, accuracy_p_test
        )
    )

    return accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test

def get_activation_c(act_c, name):
    def hook(model, input, output):
        act_c[name] = output
    return hook

def get_activation_p(act_p, name):
    def hook(model, input, output):
        act_p[name] = output
    return hook

def calActivationDiff(actCdict, actPdict, actdict, act_c, act_p):
    layer_keys = act_c.keys()
    for name in layer_keys:
        #print(act_c[name].shape)
        cleanAct = torch.linalg.norm(torch.reshape(act_c[name], (act_c[name].shape[0], -1)), dim=1, ord=2)
        perturbedAct = torch.linalg.norm(torch.reshape(act_p[name], (act_p[name].shape[0], -1)), dim=1, ord=2)
        diff_norm = torch.linalg.norm(torch.reshape(act_c[name]-act_p[name], (act_c[name].shape[0], -1)), dim=1, ord=2)
        # print(torch.abs(cleanAct-perturbedAct).shape)
        actdict[name].append(diff_norm.data.cpu())
        actCdict[name].append(cleanAct.data.cpu())
        actPdict[name].append(perturbedAct.data.cpu())
        # print(diff.data)

def calOutputDiff(outClean, outPerturbed, outDiff, outputNp, outputP):

    #print(act_c[name].shape)
    cleanOut = torch.linalg.norm(torch.reshape(outputNp, (outputNp.shape[0], -1)), dim=1, ord=2)
    perturbedOut = torch.linalg.norm(torch.reshape(outputP, (outputP.shape[0], -1)), dim=1, ord=2)
    diff_norm = torch.linalg.norm(torch.reshape(outputNp-outputP, (outputNp.shape[0], -1)), dim=1, ord=2)
    # print(torch.abs(cleanAct-perturbedAct).shape)
    outDiff.append(diff_norm.data.cpu())
    outClean.append(cleanOut.data.cpu())
    outPerturbed.append(perturbedOut.data.cpu())
    # print(diff.data)

def layerwise(act_c, act_p):
    sumLoss = 0
    MSE = nn.MSELoss()
    layer_keys = act_c.keys()
    for name in layer_keys:
        #print(MSE(act_c[name], act_p[name]))
        sumLoss += MSE(act_c[name], act_p[name])
    return sumLoss

def checkActivationDistribution(model_np_origin, model_p, trainDataLoader, attention, act_c, act_p, act_diff, act_clean, act_perturbed, epoch):

    print('Calculating activation values')

    model_np_origin.eval()
    model_p.eval()
    attention.eval()
    for batch_id, (image, label) in tqdm.tqdm(enumerate(trainDataLoader)):
        image, label = image.to(device), label.to(device)
        image_adv = attention(image)  # pylint: disable=E1102
        image_adv = image_adv.to(device)
        
        _, _ = model_np_origin(image), model_p(image_adv)
        calActivationDiff(act_clean, act_perturbed, act_diff, act_c, act_p) 

    for name, layer in model_np_origin.named_modules():
        if 'relu' in name:
            # Draw learning curve:
            # print(len(list(itertools.chain(*act_diff[name]))))
            reshapeActList = list(itertools.chain(*act_diff[name]))
            reshapeActList = [act.item() for act in reshapeActList]

            reshapeActCleanList = list(itertools.chain(*act_clean[name]))
            reshapeActCleanList = [act.item() for act in reshapeActCleanList]

            reshapeActPerturbedList = list(itertools.chain(*act_perturbed[name]))
            reshapeActPerturbedList = [act.item() for act in reshapeActPerturbedList]

            plt.hist(reshapeActList, color = 'green')
            # plt.hist(reshapeActList, bins=np.arange(min(reshapeActList), max(reshapeActList) + 10, 10), color = "skyblue", ec="skyblue")
            plt.title('L2 norm different between clean / perturbed model: {}'.format(name))
            plt.savefig(cfg.save_dir + 'NormDiff_Layer_{}_Epoch_{}.jpg'.format(name, epoch+1))
            plt.clf()

            plt.hist(reshapeActCleanList, color = 'blue')
            plt.hist(reshapeActPerturbedList, color = 'red')
            plt.legend(['Clean model', 'Perturbed model'])
            # plt.hist(reshapeActList, bins=np.arange(min(reshapeActList), max(reshapeActList) + 10, 10), color = "skyblue", ec="skyblue")
            plt.title('Activation distribution: {}'.format(name))
            plt.savefig(cfg.save_dir + 'Distribution_Layer_{}_Epoch_{}.jpg'.format(name, epoch+1))
            plt.clf()

def checkOutputDistribution(model_np_origin, model_p, trainDataLoader, attention, out_diff, out_clean, out_perturbed, epoch):

    print('Calculating output values')

    model_np_origin.eval()
    model_p.eval()
    attention.eval()
    for batch_id, (image, label) in tqdm.tqdm(enumerate(trainDataLoader)):
        image, label = image.to(device), label.to(device)
        image_adv = attention(image)  # pylint: disable=E1102
        image_adv = image_adv.to(device)
        
        output_np, output_p = model_np_origin(image), model_p(image_adv)
        calOutputDiff(out_clean, out_perturbed, out_diff, output_np, output_p) 


    # Draw learning curve:
    # print(len(list(itertools.chain(*act_diff[name]))))
    reshapeOutputList = list(itertools.chain(*out_diff))
    reshapeOutputList = [out.item() for out in reshapeOutputList]
    reshapeOutputCleanList = list(itertools.chain(*out_clean))
    reshapeOutputCleanList = [out.item() for out in reshapeOutputCleanList]
    reshapeOutputPerturbedList = list(itertools.chain(*out_perturbed))
    reshapeOutputPerturbedList = [out.item() for out in reshapeOutputPerturbedList]
    plt.hist(reshapeOutputList, color = 'green')
    # plt.hist(reshapeActList, bins=np.arange(min(reshapeActList), max(reshapeActList) + 10, 10), color = "skyblue", ec="skyblue")
    plt.title('L2 norm different between clean / perturbed model')
    plt.savefig(cfg.save_dir + 'OutputNormDiff_Epoch_{}.jpg'.format(epoch))
    plt.clf()

    plt.hist(reshapeOutputCleanList, color = 'blue')
    plt.hist(reshapeOutputPerturbedList, color = 'red')
    plt.legend(['Clean model', 'Perturbed model'])
    # plt.hist(reshapeActList, bins=np.arange(min(reshapeActList), max(reshapeActList) + 10, 10), color = "skyblue", ec="skyblue")
    plt.title('Output distribution')
    plt.savefig(cfg.save_dir + 'OutputDistribution_Epoch_{}.jpg'.format(epoch))
    plt.clf()

def transform_train(
    trainloader,
    testloader,
    arch,
    dataset,
    in_channels,
    precision,
    checkpoint_path,
    force,
    device,
    fl,
    ber,
    pos,
    seed=0,
):
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
    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    (
        model,
        checkpoint_epoch,
        model_perturbed,
        checkpoint_epoch_perturbed,
    ) = init_models_pairs(
        arch,
        in_channels,
        precision,
        True,
        checkpoint_path,
        fl,
        ber,
        pos,
        seed=seed,
    )

    print('Seed: {}'.format(seed))

    storeLoss = []

    model_np = copy.deepcopy(model) # Inference origin images

    Attention = Self_Attention()
    Attention = Attention.to(device)
    Attention.train()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, Attention.parameters()),
        lr=cfg.learning_rate,
        betas=(0.5, 0.999),
        # weight_decay=5e-4,
    )

    '''
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, Attention.parameters()),
        lr=cfg.learning_rate,
        momentum=0.9, 
        #weight_decay=5e-4
    )
    '''

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=cfg.decay
    )
    lb = cfg.lb  # Lambda


    for name, param in Attention.named_parameters():
        print("Param name: {}, grads is: {}".format(name, param.requires_grad))

    print('========== Check setting: Epoch: {}, Batch_size: {}, Lambda: {}, BitErrorRate: {}, LR: {}=========='.format(cfg.epochs, cfg.batch_size, lb, ber, cfg.learning_rate))
    print(
        "========== Start training the parameter of the input transform"
        " by using one specific perturbed model =========="
    )

    model_np, model, model_perturbed = model_np.to(device), model.to(device), model_perturbed.to(device)
    # Calculate the activate from the clean model and perturbed model
    activation_c, activation_p = {}, {}
    for name, layer in model_np.named_modules():
        if 'relu' in name:
            layer.register_forward_hook(get_activation_c(activation_c, name))
        
    for name, layer in model_perturbed.named_modules():
        if 'relu' in name:
            layer.register_forward_hook(get_activation_p(activation_p, name))

    activationDiffDict = {}
    activationCleanDict = {}
    activationPerturbedDict = {}

    from torchinfo import summary
    summary(Attention, input_size=(cfg.batch_size, 3, 32, 32))
    
    for epoch in range(cfg.epochs):
        running_loss = 0
        running_correct_orig = 0
        running_correct_p = 0

        '''
        # Reset the list for storing the activation:
        for name, layer in model_np.named_modules():
            if 'relu' in name:
                activationDiffDict[name] = []
                activationCleanDict[name] = []
                activationPerturbedDict[name] = []
        
        # Reset the list for the output distribution
        outputDiffList = []
        outputCleanList = []
        outputPerturbedList = []

        #if epoch % 2 == 0:
        #    checkActivationDistribution(model_np, model_perturbed, trainloader, Attention, activation_c, activation_p, activationDiffDict, activationCleanDict, activationPerturbedDict, epoch)
        #    # checkOutputDistribution(model_np, model_perturbed, trainloader, Attention, outputDiffList, outputCleanList, outputPerturbedList, epoch)
        '''

        Attention.train()
              
        # For each epoch, we will use N perturbed model for training.
        for batch_id, (image, label) in tqdm.tqdm(enumerate(trainloader)):
            total_loss = 0  
            image, label = image.to(device), label.to(device)
            image_adv = Attention(image)  # pylint: disable=E1102
            image_adv = image_adv.to(device)

            model_np.eval()
            model.eval()
            model_perturbed.eval()

            _, out, out_biterror = model_np(image), model(image_adv), model_perturbed(image_adv)  # pylint: disable=E1102

            loss_orig, pred_orig = compute_loss(out, label)
            loss_p, pred_p = compute_loss(out_biterror, label)

            running_correct_orig += torch.sum(pred_orig == label.data).item()
            running_correct_p += torch.sum(pred_p == label.data).item()
            
            # Calculate the total loss. 
            if cfg.layerwise:
                #print(layerwise(activation_c, activation_c))
                loss = loss_orig + lb * (loss_p + layerwise(activation_c, activation_p))
            else:
                loss = loss_orig + lb * loss_p

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()        

        accuracy_orig = running_correct_orig / len(trainloader.dataset)
        accuracy_p = running_correct_p / (len(trainloader.dataset))
        print(
            "For epoch: {}, loss: {:.6f}, accuracy for clean model:"
            "{:.5f}, accuracy for perturbed model: {:.5f}".format(
                epoch + 1,
                running_loss,
                accuracy_orig,
                accuracy_p,
            )
        )

        storeLoss.append(running_loss)

        #if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.epochs:
        #    # Saving the result of the generator!
        #    torch.save(Attention,
        #        cfg.save_dir + 'Single_AttentionV1_arch_{}_LR{}_E_{}_ber_{}_lb_{}_NOWE_{}.pt'.format(arch, cfg.learning_rate, cfg.epochs, ber, lb, epoch+1))

    '''
    # For the final testing!
    # Reset the list for storing the activation:
    for name, layer in model_np.named_modules():
        if 'relu' in name:
            activationDiffDict[name] = []
            activationCleanDict[name] = []
            activationPerturbedDict[name] = []

    # checkActivationDistribution(model_np, model_perturbed, trainloader, Attention, activation_c, activation_p, activationDiffDict, activationCleanDict, activationPerturbedDict, cfg.epochs)

    # Reset the list for the output distribution
    outputDiffList = []
    outputCleanList = []
    outputPerturbedList = []

    # checkOutputDistribution(model_np, model_perturbed, trainloader, Attention, outputDiffList, outputCleanList, outputPerturbedList, cfg.epochs)
    '''

    # Draw learning curve:
    plt.plot([e+1 for e in range(cfg.epochs)], storeLoss)
    plt.title('Learning Curve')
    plt.savefig(cfg.save_dir + 'Learning_Curve.jpg')

    print('========== Start checking the accuracy with different perturbed model ==========')

    model, _, model_perturbed, _ = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=seed)
    model, model_perturbed = model.to(device), model_perturbed.to(device)
    model.eval()
    model_perturbed.eval()
    Attention.eval()

    # Without using transform
    accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Attention, device, use_transform=False)

    # With input transform
    accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Attention, device, use_transform=True)

  