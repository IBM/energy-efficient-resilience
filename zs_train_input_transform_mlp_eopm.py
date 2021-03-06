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
import random
import tqdm
from collections import OrderedDict

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


class Program(nn.Module):
    """
    Apply reprogramming.
    """

    def __init__(self, cfg):
        super(Program, self).__init__()
        self.cfg = cfg
        self.dense = nn.Linear(self.cfg.channels*self.cfg.h1*self.cfg.w1, self.cfg.channels*self.cfg.h1*self.cfg.w1)
        self.P = None
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.tanh_fn = torch.nn.Tanh()
        self.init_perturbation()

    def init_perturbation(self):
        init_p = torch.zeros((self.cfg.channels, self.cfg.h1, self.cfg.w1))
        self.P = Parameter(init_p, requires_grad=True)    

    def forward(self, image):
        x = image.data.clone()

        x_flatten = self.flatten(x)
        x_mlp = self.dense(x_flatten)
        x_mlp = torch.reshape(x_mlp, (-1, 3, 32, 32))
        
        out = x + self.tanh_fn(self.P) + self.relu(x_mlp)
        out = torch.clamp(out, min=-1, max=1)
        return out


def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(labels.size(0))  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds

def init_dict(model_transform, grad_dict):
    for param_name, param_weight in model_transform.named_parameters():
        if param_weight.requires_grad:
            grad_dict[param_name] = 0

def sum_grads(model_transform, grad_dict):
    for param_name, param_weight in model_transform.named_parameters():
        if param_weight.requires_grad:
            grad_dict[param_name] += param_weight.grad 

def mean_grads(grad_dict, nums):
    param_names = grad_dict.keys()
    for param_name in param_names:
        grad_dict[param_name] /= nums

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
        #weight_decay=5e-4
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=200, gamma=cfg.decay
    )

    lb = cfg.lb  # Lambda 

    for name, param in Pg.named_parameters():
        print("Param name: {}, grads is: {}".format(name, param.requires_grad))

    print('========== Check setting: Epoch: {}, Batch_size: {}, N perturbed models: {}, Lambda: {}, BitErrorRate: {}, LR: {} =========='.format(cfg.epochs, cfg.batch_size, cfg.N, lb, ber, cfg.learning_rate))

    print(
        "========== Start training the parameter"
        " of the input transform by using EOT attack =========="
    )
    for epoch in range(cfg.epochs):
        print('********** Epoch : {} **********'.format(epoch))
        running_loss = 0
        running_correct_orig = 0
        running_correct_p = 0
        loss = 0

        # For each epoch, we will use N perturbed model for training.
        for batch_id, (image, label) in tqdm.tqdm(enumerate(trainloader)):
            
            gradDict = OrderedDict()
            init_dict(Pg, gradDict)

            image, label = image.to(device), label.to(device)
            
            for k in range(cfg.N):
                
                image_adv = Pg(image)  # pylint: disable=E1102, Prevent "Trying to backward through the graph a second time" error!
                image_adv = image_adv.to(device)

                # Random test
                if cfg.totalRandom:
                    j = random.randint(0, cfg.randomRange)
                else:
                    j = k

                # print("*** For using the {}-th perturbed model ***".format(j))
                model, _, model_perturbed, _ = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=j)
                    
                model, model_perturbed = model.to(device), model_perturbed.to(device)
                model.eval()
                model_perturbed.eval()

                # Inference the clean model and perturbed model
                out = model(image_adv)  # pylint: disable=E1102
                out_biterror = model_perturbed(image_adv)  # pylint: disable=E1102

                # Compute the loss for clean model and perturbed model
                loss_orig, pred_orig = compute_loss(out, label)
                loss_p, pred_p = compute_loss(out_biterror, label)                    
                    
                # Keep the running accuracy of clean model and perturbed model.
                running_correct_orig += torch.sum(pred_orig == label.data).item()
                running_correct_p += torch.sum(pred_p == label.data).item()     
            
                # Calculate the total loss. For easier codding, we inference clean model for each j, but it is still the same objective function of ver2.
                loss = loss_orig + lb * loss_p
                running_loss += loss.item()  

                # Calculate the gradients
                optimizer.zero_grad()
                loss.backward()
                
                # Sum all of the gradients
                sum_grads(Pg, gradDict)

            # Average the gradients
            mean_grads(gradDict, cfg.N)

            # Set gradients back to P
            for param_name, param_weight in Pg.named_parameters():
                param_weight.grad = gradDict[param_name]

            # Apply gradients by optimizer            
            optimizer.step()
            lr_scheduler.step()
            
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

    for i in range(50000, 50010):
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
        '{}/arch_{}_LR_{}_E_{}_gradsmean_N_{}_ber_{}_lb_{}.pt'.format(cfg.save_dir, arch, cfg.learning_rate, cfg.epochs, cfg.N, ber, lb))