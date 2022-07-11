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
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import copy
import os

from config import cfg
from models import init_models_pairs, create_faults
from models.generator import *
import faultsMap as fmap


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-20

    
def check_dir(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(labels.size(0))  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds

def accuracy_checking(
    model_orig, model_p, trainloader, testloader, gen, device, use_transform=False
):
    """
      Calculating the accuracy with given clean model and perturbed model.
      :param model_orig: Clean model.
      :param model_p: Perturbed model.
      :param trainloader: The loader of training data.
      :param testloader: The loader of testing data.
      :param gen: The object of the Generator.
      :param device: Specify GPU usage.
      :use_transform: Should apply input transformation or not.
    """
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
            x_adv = gen(x)
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
            x_adv = gen(x)
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

def add_into_weights(model_p, diff):
    names_diff = diff.keys()
    for name, param in model_p.named_parameters():
        if name in names_diff:
            param.data = param.data + diff[name]            

def diff_in_weight(model_orig, model_p):
    diff_dict = OrderedDict()
    for(old_k, old_w), (new_k, new_w) in zip(model_orig.named_parameters(), model_p.named_parameters()):
        if old_w.requires_grad:
            diff_w = new_w - old_w
            diff_dict[old_k] = diff_w
    return diff_dict

def quantization(model_weights, precision=8):
    max_val = torch.max(torch.abs(model_weights))
    delta = max_val / (2 ** (precision - 1) - 1)
    return delta

def applyBitAttack(clean, perturbed):
    return (clean ^ (2 ** torch.floor(torch.log2(clean ^ perturbed))).to(torch.uint8)).to(torch.uint8)

def bitErrorAttack(allParamClean, allParamPerturb, allGrads, allDeleta, topK, alpha, precision, device):
    # sumDiff = 0
    mask = torch.zeros(allParamClean.shape).to(device)
    mask[topK] = 1
    changeParamTmp = allParamPerturb + alpha * (mask * allGrads) # Apply gradients ascent on perturbed weights, and set it temp variable.
    changeParamTmp = torch.clamp(changeParamTmp, min=-128*allDeleta, max=127*allDeleta) # Clamp the perturbed weights into correct range.    
    cleanParam_q = torch.round((allParamClean/ allDeleta)) # Quantized the weights from the clean model by using symmetric-signed.
    cleanParam_q = cleanParam_q.to(torch.int8)
    changeParam_q = torch.round((changeParamTmp / allDeleta)) # Quantized the weights of the perturbed model by using symmetric-signed.
    changeParam_q = changeParam_q.to(torch.int8)

    # for idx in topK:
    # Change the perturbed weights into byte. Reference: https://stackoverflow.com/questions/72851439/fast-computation-for-changing-the-leftmost-different-bit
    bit_tmp = applyBitAttack(cleanParam_q.to(torch.uint8), changeParam_q.to(torch.uint8))
    bit_tmp = bit_tmp.to(torch.int8) 
    allParamPerturb = bit_tmp * allDeleta # Dequantized the perturbed weights and set the results to perturbed models.
    
    # print('-----------------------------------------')
    # print("Origin bit_clean: {}".format(bin(cleanParam_q[topK[0]].to(torch.uint8))))
    # print("Origin bit_tmp: {}".format(bin(changeParam_q[topK[0]].to(torch.uint8))))
    # print("Afrer change bit_tmp: {}".format(bin(bit_tmp[topK[0]].to(torch.uint8))))
    # print('')
    # print("Origin bit_clean: {}".format(bin(cleanParam_q[topK[200]].to(torch.uint8))))
    # print("Origin bit_tmp: {}".format(bin(changeParam_q[topK[200]].to(torch.uint8))))
    # print("Afrer change bit_tmp: {}".format(bin(bit_tmp[topK[200]].to(torch.uint8))))
    # print('')
    # print("Origin bit_clean: {}".format(bin(cleanParam_q[5].to(torch.uint8))))
    # print("Origin bit_tmp: {}".format(bin(changeParam_q[5].to(torch.uint8))))
    # print("Afrer change bit_tmp: {}".format(bin(bit_tmp[5].to(torch.uint8))))
    # print('')
    # print(5 in topK)
    # print("different: {}".format(allParamPerturb[idx] - bit_tmp * allDeleta[idx]))
    # sumDiff += torch.sum(allParamPerturb[idx] - bit_tmp * allDeleta[idx]).abs()
        
    
    # print("Inner diff: {}".format(sumDiff))
        
    return allParamPerturb


def pgd(model_origin, model_p, alpha, precision, device):

    """
    Apply PGD attack on the model weights.
    Only K weights can be attacked.
    And only one bit can be attacked on each perturbed weights.
    So, choose top-K gradients to update the weights.
    
    :param model_origin: The clean model.
    :param model_p: The perturbed model.
    :param alpha: The step update the perturbed weights.
    :param precision: An int. The number of bits would be used to quantize the model.
    :param device: Specify GPU usage.
    """
    
    allParamClean = None # Store the flatten clean weights.
    allParam = None # Store the flatten perturbed weights.
    allGrads = None # Store the gradients that will be used to update the weights of the perturbed model.
    allDeleta = None # Store the quantization scale of the each weights. deleta = max_weights / (2 ** (precision - 1) - 1)
    storeParamNum = [] # Store how many weights in each layer. This is for reshaping the weights.
    
    # Store the flatten weights and gradient first:
    for (name_origin, param_origin), (name_p, param_p) in zip(model_origin.named_parameters(), model_p.named_parameters()):
        if not 'bn' in name_p:
            delta = quantization(param_p.data)
            ParamNum = torch.numel(param_p.grad.data)
            storeParamNum.append(ParamNum)
            flattenParamClean = torch.flatten(param_origin.data)
            flattenParam = torch.flatten(param_p.data)
            flattenGrads = torch.flatten(param_p.grad.data)
            if allParam == None: # Initial the storage
                allParamClean = flattenParamClean
                allParam = flattenParam
                allGrads = flattenGrads
                allDeleta = delta.expand(ParamNum)
            else:
                allParamClean = torch.concat((allParamClean, flattenParamClean))
                allParam = torch.concat((allParam, flattenParam))
                allGrads = torch.concat((allGrads, flattenGrads))
                allDeleta = torch.concat((allDeleta, delta.expand(ParamNum)))

    # Find top K gradients:
    topKVal, topKIdx = torch.topk(allGrads.abs(), 10000)
    # Apply attack on weights
    allParam = bitErrorAttack(allParamClean, allParam, allGrads, allDeleta, topKIdx, alpha, precision, device)

    # Replace the new weight into perturbed models.
    countNum = 0
    weightNum = 0
    for name_p, param_p in model_p.named_parameters():
        if not 'bn' in name_p:
            param_p.data = torch.reshape(allParam[weightNum: weightNum + storeParamNum[countNum]], param_p.shape)
            weightNum += storeParamNum[countNum]
            countNum += 1

        

    # print(param.grad.abs().topk(5))
    # print(param.grad.abs().shape)
    # print(torch.topk(allParam.abs(), 5))
    # print(allParam[0].grad)
    # print( torch.reshape(allParam, (len()))
        
# For layerwise training
def get_activation_c(act_c, name):
    def hook(model, input, output):
        act_c[name] = output
    return hook

def get_activation_p(act_p, name):
    def hook(model, input, output):
        act_p[name] = output
    return hook

def layerwise(act_c, act_p):
    sumLoss = 0
    MSE = nn.MSELoss()
    layer_keys = act_c.keys()
    for name in layer_keys:
        #print(MSE(act_c[name], act_p[name]))
        sumLoss += MSE(act_c[name], act_p[name])
    return sumLoss
                

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

    check_dir([cfg.save_dir, cfg.save_dir_curve])

    storeLoss = []

    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True


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
        
    Gen = Gen.to(device)
    Gen.train()

    # Using Adam:
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, Gen.parameters()),
        lr=cfg.learning_rate,
        betas=(0.5, 0.999),
        # weight_decay=5e-4,
    )

    # Using SGD:
    #optimizer = torch.optim.SGD(
    #    filter(lambda p: p.requires_grad, Pg.parameters()),
    #    lr=cfg.learning_rate,
    #    momentum=0.9, 
    #    weight_decay=1e-4
    #)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=200, gamma=cfg.decay
    )
    lb = cfg.lb  # Lambda 
    # cfg.alpha = (1 / cfg.PGD_STEP) 

    for name, param in Gen.named_parameters():
        print("Param name: {}, grads is: {}".format(name, param.requires_grad))

    print('========== Check setting: Epoch: {}, Batch_size: {}, Lambda: {}, BitErrorRate: {}, G: {} =========='.format(cfg.epochs, cfg.batch_size, lb, ber, cfg.G))
    print('========== Check PGD setting: STEP: {}, LR_p: {}, LR_a: {} =========='.format(cfg.PGD_STEP, cfg.learning_rate, cfg.alpha))
    print('========== Check folder: save_dir: {}, save_dir_curve: {} =========='.format(cfg.save_dir, cfg.save_dir_curve))
    print('========== Layerwise Training: {}, Random Training: {}'.format(cfg.layerwise, cfg.totalRandom))
    print("========== Start training the parameter of the input transform by using Adversarial Training ==========")

    (model, _, _, _) = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=0, dataset=dataset) # Create clean model.
    
    # EPSILON = ber * 511 #(256+128+64+32+16+8+4+2+1)
    # EPSILON = 0.01 * 128 #(256+128+64+32+16+8+4+2+1)

    learning_curve_loss = []
 
    for epoch in range(cfg.epochs):
        running_loss = 0
        running_correct_orig = 0
        running_correct_p = 0
              
        # For each epoch, we will use N perturbed model for training.
        for batch_id, (image, label) in tqdm.tqdm(enumerate(trainloader)):
            
            model_perturbed = copy.deepcopy(model)
            model.eval()
            model_perturbed.train()
            Gen.eval()

            # Apply AWP with PGD!
            for i in range(cfg.PGD_STEP): 
                loss_pgd = 0
                model, model_perturbed = model.to(device), model_perturbed.to(device)
                for param in model_perturbed.parameters():
                    param.requires_grad = True
                for param in Gen.parameters():
                    param.requires_grad = False
                Gen.zero_grad()
                image, label = image.to(device), label.to(device)
                image_adv = Gen(image)  # pylint: disable=E1102, Prevent "Trying to backward through the graph a second time" error!
                image_adv = image_adv.to(device)
                out_perturb = model_perturbed(image_adv)
                loss_pgd, pred_pgd = compute_loss(out_perturb, label)
                loss_pgd.backward()
                pgd(model, model_perturbed, cfg.alpha, precision, device)
                # print(loss_pgd)
                # sumDiff = 0
                # for param_a, param_p in zip(model.parameters(), model_perturbed.parameters()):
                #     sumDiff += torch.sum(param_p - param_a).abs()
                #     print(param_p - param_a)
                # print("Outer diff: {}".format(sumDiff))

            #print('------------------------------------------------------')
            # Start to update parameter P
            for param in Gen.parameters():
                param.requires_grad = True

            
            model, model_perturbed = model.to(device), model_perturbed.to(device)
            model.eval()
            model_perturbed.eval()
            Gen.train()

            loss = 0
            image, label = image.to(device), label.to(device)
            image_adv = Gen(image)  # pylint: disable=E1102, Prevent "Trying to backward through the graph a second time" error!
            image_adv = image_adv.to(device)

            # model_np = copy.deepcopy(model) # Inference origin images
            # model_np, model, model_perturbed = model_np.to(device), model.to(device), model_perturbed.to(device)
            # model_np.eval()
            
            

            # Calculate the activate from the clean model and perturbed model
            #activation_c, activation_p = {}, {}
            #for name, layer in model_np.named_modules():
            #    if 'relu' in name:
            #        layer.register_forward_hook(get_activation_c(activation_c, name))
                
            #for name, layer in model_perturbed.named_modules():
            #    if 'relu' in name:
            #        #print(name)
            #        layer.register_forward_hook(get_activation_p(activation_p, name))

            # _, out, out_perturb = model_np(image), model(image_adv), model_perturbed(image_adv)
            out, out_perturb = model(image_adv), model_perturbed(image_adv)

            # Compute the loss for clean model and perturbed model
            loss_orig, pred_orig = compute_loss(out, label)   
            loss_perturb, pred_perturb = compute_loss(out_perturb, label)            
                
            # Keep the running accuracy of clean model and perturbed model.
            running_correct_orig += torch.sum(pred_orig == label.data).item()
            running_correct_p += torch.sum(pred_perturb == label.data).item() 
        
            # Calculate the total loss. 
            if cfg.layerwise:
                #print(layerwise(activation_c, activation_p))
                loss = loss_orig + lb * (loss_perturb + layerwise(activation_c, activation_p))
            else:
                loss = loss_orig + lb * loss_perturb

            running_loss += loss.item()
            # Apply gradients by optimizer            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # for param in Gen.parameters():
            #     print(param.grad)

        # learning_curve_loss.append(running_loss) 

        # Keep the running accuracy of clean model and perturbed model for all mini-batch.
        accuracy_orig = running_correct_orig / (len(trainloader.dataset))
        accuracy_p = running_correct_p / (len(trainloader.dataset))
        print(
            "For epoch: {}, loss: {:.6f}, accuracy for clean model: "
            "{:.5f}, accuracy perturbed model: {:.5f}".format(
                epoch + 1,
                running_loss,
                accuracy_orig,
                accuracy_p,
            )
        )

        storeLoss.append(running_loss)

        # same: use (epsilon*delta), different: only fix alpha
        if (epoch + 1) % 50 == 0 or (epoch + 1) == cfg.epochs:
            # Saving the result of the generator!
            torch.save(Gen.state_dict(),
                cfg.save_dir + 'Adversarial_GeneratorV1_{}_arch_{}_LR_a{}_p{}_E_{}_PGD_{}_ber_{}_lb_{}_NOWE_{}_bit_10000.pt'.format(dataset, arch, cfg.alpha, cfg.learning_rate, cfg.epochs, cfg.PGD_STEP, ber, lb, epoch+1))

    # Draw learning curve:
    plt.plot([e+1 for e in range(cfg.epochs)], storeLoss)
    plt.title('Learning Curve')
    plt.savefig(cfg.save_dir + 'Adversarial_GeneratorV1_{}_arch_{}_LR_a{}_p{}_E_{}_PGD_{}_ber_{}_lb_{}_NOWE_{}_bit_10000_Learning_Curve.jpg'.format(dataset, arch, cfg.alpha, cfg.learning_rate, cfg.epochs, cfg.PGD_STEP, ber, lb, epoch+1))


    # -------------------------------------------------- Inference --------------------------------------------------


    print('========== Start checking the accuracy with different perturbed model: Adversarial bit error mode ==========')

    cfg.replaceWeight = False

    (model, _, _, _) = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=0, dataset=dataset) # Create clean model.

    model_perturbed = copy.deepcopy(model)
    model.eval()
    model_perturbed.train()

    # Apply AWP with PGD!
    for i in range(cfg.PGD_STEP): 
        loss_pgd = 0
        model, model_perturbed = model.to(device), model_perturbed.to(device)
        for param in model_perturbed.parameters():
            param.requires_grad = True
        Gen.zero_grad()
        image, label = image.to(device), label.to(device)
        image_adv = Gen(image)  # pylint: disable=E1102, Prevent "Trying to backward through the graph a second time" error!
        out_perturb = model_perturbed(image_adv)
        loss_pgd, pred_pgd = compute_loss(out_perturb, label)
        loss_pgd.backward()
        pgd(model, model_perturbed, cfg.alpha, precision, device)
        #print(loss_pgd)
        #for param_a, param_p in zip(model.parameters(), model_perturbed.parameters()):
        #    print(param_p - param_a)


    accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Gen, device, use_transform=False)
    accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Gen, device, use_transform=True)


    print('========== Start checking the accuracy with different perturbed model: Random bit error mode ==========')
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

    model, _, model_perturbed, _ = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl, ber, pos, dataset=dataset)
    model, model_perturbed = model.to(device), model_perturbed.to(device)
    for i in range(cfg.beginSeed, cfg.endSeed):
        print(' ********** For seed: {} ********** '.format(i))
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