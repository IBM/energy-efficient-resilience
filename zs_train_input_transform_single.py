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
import numpy as np
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
        self.num_classes = 10

        self.P = None
        self.tanh_fn = torch.nn.Tanh()
        self.init_perturbation()

    # Initialize Perturbation
    def init_perturbation(self):
        init_p = torch.zeros((self.cfg.channels, self.cfg.h1, self.cfg.w1))
        self.P = Parameter(init_p, requires_grad=True)

    def forward(self, image):
        x = image.data.clone()
        x_adv = x + self.tanh_fn(self.P)
        x_adv = torch.clamp(x_adv, min=-1, max=1)
        return x_adv


def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(
        labels.size(0)
    )  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds


def accuracy_checking(
    model_orig, model_p, trainloader, testloader, pg, device, use_transform=False
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

    model_np = copy.deepcopy(model) # Inference origin images

    Pg = Program(cfg)
    Pg = Pg.to(device)
    Pg.train()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, Pg.parameters()),
        lr=cfg.learning_rate,
        betas=(0.5, 0.999),
        weight_decay=5e-4,
    )

    '''
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, Pg.parameters()),
        lr=cfg.learning_rate,
        momentum=0.9, 
        #weight_decay=5e-4
    )
    '''

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=200, gamma=cfg.decay
    )
    lb = cfg.lb  # Lambda


    for name, param in Pg.named_parameters():
        print("Param name: {}, grads is: {}".format(name, param.requires_grad))

    print('========== Check setting: Epoch: {}, Batch_size: {}, Lambda: {}, BitErrorRate: {}, LR: {}=========='.format(cfg.epochs, cfg.batch_size, lb, ber, cfg.learning_rate))
    print(
        "========== Start training the parameter of the input transform"
        " by using one specific perturbed model =========="
    )

    model_np, model, model_perturbed = model_np.to(device), model.to(device), model_perturbed.to(device)

    for epoch in range(cfg.epochs):
        running_loss = 0
        running_correct_orig = 0
        running_correct_p = 0
              
        # For each epoch, we will use N perturbed model for training.
        for batch_id, (image, label) in enumerate(trainloader):
            total_loss = 0  
            image, label = image.to(device), label.to(device)
            image_adv = Pg(image)  # pylint: disable=E1102
            image_adv = image_adv.to(device)

            model_np.eval()
            model.eval()
            model_perturbed.eval()

            # Calculate the activate from the clean model and perturbed model
            activation_c, activation_p = {}, {}
            for name, layer in model_np.named_modules():
                layer.register_forward_hook(get_activation_c(activation_c, name))
                
            for name, layer in model_perturbed.named_modules():
                layer.register_forward_hook(get_activation_p(activation_p, name))

            _, out, out_biterror = model_np(image), model(image_adv), model_perturbed(image_adv)  # pylint: disable=E1102

            loss_orig, pred_orig = compute_loss(out, label)
            loss_p, pred_p = compute_loss(out_biterror, label)

            running_correct_orig += torch.sum(pred_orig == label.data).item()
            running_correct_p += torch.sum(pred_p == label.data).item()
            
            # Calculate the total loss. 
            if cfg.layerwise:
                #print(layerwise(activation_c, activation_c))
                loss = loss_orig + lb * layerwise(activation_c, activation_p)
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


    print('========== Start checking the accuracy with different perturbed model ==========')

    model, _, model_perturbed, _ = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=seed)
    model, model_perturbed = model.to(device), model_perturbed.to(device)
    model.eval()
    model_perturbed.eval()
    Pg.eval()

    # Without using transform
    accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Pg, device, use_transform=False)

    # With input transform
    accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Pg, device, use_transform=True)

  