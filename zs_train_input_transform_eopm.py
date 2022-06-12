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
import random
import tqdm
import copy

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


class Program(nn.Module):
    """
      Apply reprogramming.
    """

    def __init__(self, cfg):
        super(Program, self).__init__()
        self.cfg = cfg
        self.P = None
        self.tanh_fn = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
        self.init_perturbation()

    # Initialize Perturbation
    def init_perturbation(self):
        init_p = torch.zeros((self.cfg.channels, self.cfg.h1, self.cfg.w1))
        self.P = Parameter(init_p, requires_grad=True)
        # self.P = torch.nn.init.xavier_uniform_(Parameter(init_p, requires_grad=True))
        # self.P = torch.nn.init.uniform_(Parameter(init_p, requires_grad=True), a=-1.0, b=1.0)
        # self.P = Parameter(torch.nn.init.kaiming_uniform_(init_p), requires_grad=True)

    def forward(self, image):
        x = image.data.clone()
        # x_adv = x + self.tanh_fn(self.P)
        # x_adv = x + torch.clamp(self.P, min=-1, max=1)
        # x_adv = torch.clamp(x_adv, min=-1, max=1)
        # x_adv = x + torch.clamp(self.dropout(self.P), min=-1, max=1)
        # x_adv = torch.clamp(x_adv, min=-1, max=1)
        x_adv = torch.tanh(0.5 * (torch.log(1 + x + 1e-15) - torch.log(1 - x + 1e-15)) + self.dropout(self.P))
        return x_adv


def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(labels.size(0))  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds


def accuracy_checking(
    model_orig, model_p, trainloader, testloader, pg, device, use_transform=False
):
    """
      Calculating the accuracy with given clean model and perturbed model.
      :param model_orig: Clean model.
      :param model_p: Perturbed model.
      :param trainloader: The loader of training data.
      :param testloader: The loader of testing data.
      :param pg: The object of the Pg class.
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
            x_adv = pg(x)
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
            x_adv = pg(x)
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

def layerwise(act_c, act_p):
    sumLoss = 0
    MSE = nn.MSELoss()
    layer_keys = act_c.keys()
    for name in layer_keys:
        #print(MSE(act_c[name], act_p[name]))
        sumLoss += MSE(act_c[name], act_p[name])
    return sumLoss
 
def imgMSE(img_o, img_p):
    MSE = nn.MSELoss()
    print("Image MSE: {}".format(MSE(img_o, img_p)))

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

    storeLoss = []

    Pg = Program(cfg)
    Pg = Pg.to(device)
    Pg.train()

    # Using Adam:
    #optimizer = torch.optim.Adam(
    #    filter(lambda p: p.requires_grad, Pg.parameters()),
    #    lr=cfg.learning_rate,
    #    betas=(0.5, 0.999),
    #    weight_decay=5e-4,
    #)

    # Using SGD:
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, Pg.parameters()),
        lr=cfg.learning_rate,
        momentum=0.9, 
        # weight_decay=1e-4,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=200, gamma=cfg.decay
    )
    lb = cfg.lb  # Lambda 

    for name, param in Pg.named_parameters():
        print("Param name: {}, grads is: {}".format(name, param.requires_grad))

    print('========== Check setting: Epoch: {}, Batch_size: {}, N perturbed models: {}, Lambda: {}, BitErrorRate: {}, LR: {}=========='.format(cfg.epochs, cfg.batch_size, cfg.N, lb, ber, cfg.learning_rate))
    print('========== Layerwise Training: {}, Random Training: {}'.format(cfg.layerwise, cfg.totalRandom))

    print(
        "========== Start training the parameter"
        " of the input transform by using EOT attack =========="
    )

    for epoch in range(cfg.epochs):
        running_loss = 0
        running_correct_orig = 0
        running_correct_p = 0
        each_c_pred = [0] * cfg.N
        each_p_pred = [0] * cfg.N
              
        # For each epoch, we will use N perturbed model for training.
        for batch_id, (image, label) in tqdm.tqdm(enumerate(trainloader)):
            total_grads = 0  
            image, label = image.to(device), label.to(device)
            
            for k in range(cfg.N):
                
                loss = 0

                image_adv = Pg(image)  # pylint: disable=E1102, Prevent "Trying to backward through the graph a second time" error!
                image_adv = image_adv.to(device)
                
                # Random test
                if cfg.totalRandom:
                    j = random.randint(0, cfg.randomRange)
                else:
                    j = k

                model, _, model_perturbed, _ = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=j)
                if cfg.layerwise:
                    model_np = copy.deepcopy(model) # Inference origin images
                    model_np, model, model_perturbed = model_np.to(device), model.to(device), model_perturbed.to(device)
                    model_np.eval()
                else:
                    model, model_perturbed = model.to(device), model_perturbed.to(device)
                
                model.eval()
                model_perturbed.eval()

                # Calculate the activate from the clean model and perturbed model
                if cfg.layerwise:
                    activation_c, activation_p = {}, {}
                    for name, layer in model_np.named_modules():
                        if 'relu' in name:
                            layer.register_forward_hook(get_activation_c(activation_c, name))

                    for name, layer in model_perturbed.named_modules():
                        if 'relu' in name:
                            #print(name)
                            layer.register_forward_hook(get_activation_p(activation_p, name))

                # Inference the clean model and perturbed model
                out_biterror_without_p = model_perturbed(image) 
                _, pred_without_p = torch.max(out_biterror_without_p, 1)
                each_c_pred[k] += torch.sum(pred_without_p == label.data).item()

                if cfg.layerwise:
                    _ = model_np(image)

                out = model(image_adv)  # pylint: disable=E1102
                out_biterror = model_perturbed(image_adv)  # pylint: disable=E1102   
                              
                # Compute the loss for clean model and perturbed model
                loss_orig, pred_orig = compute_loss(out, label)
                loss_p, pred_p = compute_loss(out_biterror, label)     
                #print(pred_p)   

                each_p_pred[k] += torch.sum(pred_p == label.data).item()     
                    
                # Keep the running accuracy of clean model and perturbed model.
                running_correct_orig += torch.sum(pred_orig == label.data).item()
                running_correct_p += torch.sum(pred_p == label.data).item() 

                # Calculate the total loss. 
                if cfg.layerwise:
                    #print(layerwise(activation_c, activation_p))
                    loss = loss_orig + lb * (loss_p + layerwise(activation_c, activation_p))
                else:
                    loss = loss_orig + lb * loss_p

                # Keep the overal loss for whole batches
                running_loss += loss.item()  

                # Calculate the gradients
                optimizer.zero_grad()
                loss.backward()
                
                #for param in model_perturbed.parameters():
                #    print(param.grad)

                # print('{}. Grad norm: {}'.format(k, torch.linalg.norm(torch.flatten(Pg.P.grad), dim=0, ord=2)))
                # g_norm = Pg.P.grad / torch.linalg.norm(torch.flatten(Pg.P.grad), dim=0, ord=2)
                # print('{}. Grad normalization norm: {}'.format(k, torch.linalg.norm(torch.flatten(g_norm), dim=0, ord=2)))
                # print('{}. Grad normalization norm: {}'.format(k, g_norm))
                
                # Sum all of the gradients
                total_grads += Pg.P.grad
                #total_grads += g_norm
                
            # imgMSE(image, image_adv)
            # Average the gradients
            mean_grads = total_grads / cfg.N

            # Set gradients back to P
            for param in Pg.parameters():
                param.grad = mean_grads

            # Apply gradients by optimizer to parameter           
            optimizer.step()
            lr_scheduler.step()
            
        print('P 2-norm: {}, MSE: {}'.format(torch.linalg.norm(torch.flatten(Pg.P), dim=0, ord=2), imgMSE(image, image_adv)))
        print('Each pred w/o transformation: {}'.format([x/len(trainloader.dataset) for x in each_c_pred]))
        print('Each pred with transformation: {}'.format([x/len(trainloader.dataset) for x in each_p_pred]))
            
        # Keep the running accuracy of clean model and perturbed model for all mini-batch.
        accuracy_orig = running_correct_orig / (len(trainloader.dataset) * cfg.N)
        accuracy_p = running_correct_p / (len(trainloader.dataset) * cfg.N)
        print(
            "For epoch: {}, loss: {:.6f}, accuracy for {} clean model:"
            "{:.5f}, accuracy for {} perturbed model: {:.5f}".format(
                epoch + 1,
                running_loss / cfg.N,
                cfg.N,
                accuracy_orig,
                cfg.N,
                accuracy_p,
            )
        )

        storeLoss.append(running_loss / cfg.N)

    # Draw learning curve:
    plt.plot([e+1 for e in range(cfg.epochs)], storeLoss)
    plt.title('Learning Curve')
    plt.savefig('result.jpg')

    print('========== Start checking the accuracy with different perturbed model ==========')
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

    for i in range(cfg.beginSeed, cfg.endSeed):
        print(' ********** For seed: {} ********** '.format(i))
        model, _, model_perturbed, _ = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=i)
        model, model_perturbed = model.to(device), model_perturbed.to(device),

        model.eval()
        model_perturbed.eval()
        Pg.eval()

        # Without using transform
        accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Pg, device, use_transform=False)
        accuracy_orig_train_list.append(accuracy_orig_train)
        accuracy_p_train_list.append(accuracy_p_train)
        accuracy_orig_test_list.append(accuracy_orig_test)
        accuracy_p_test_list.append(accuracy_p_test)

        # With input transform
        accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Pg, device, use_transform=True)
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

    # Saving the result of the parameter!
    torch.save({'Reprogrammed Perturbation': Pg.P},
        cfg.save_dir + 'EOPM_arch_{}_LR_p{}_E_{}_ber_{}_lb_{}_Normal_initZero.pt'.format(arch, cfg.learning_rate, cfg.epochs, ber, lb))
