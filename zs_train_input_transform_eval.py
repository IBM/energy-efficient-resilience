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

from config import cfg
from models import init_models_pairs
import numpy as np
import copy


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-20
PGD_STEP = 2


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

        # self.temperature = self.cfg.temperature  # not being used yet

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

def transform_eval(
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

    Pg = Program(cfg)
    Pg.P = torch.load(cfg.P_PATH)['Reprogrammed Perturbation']
    Pg.to(device)
    Pg.eval()
    print('Successfully load input transformation parameter!')

    if(cfg.testing_mode == 'random_bit_error'):
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
    
        for i in range(50001, 50005):
            print(' ********** For seed: {} ********** '.format(i))
            (model, checkpoint_epoch, model_perturbed, checkpoint_epoch_perturbed) = init_models_pairs( 
                        arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=i)
            model, model_perturbed = model.to(device), model_perturbed.to(device),
            model.eval()
            model_perturbed.eval()
    
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
    
    