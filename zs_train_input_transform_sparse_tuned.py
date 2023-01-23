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
from models import init_models
import numpy as np

import torchvision
import torchvision.transforms as transforms

import random, pprint

import pdb

import zs_hooks_stats as stats

torch.manual_seed(0)
device = 'cuda'

layer_loss = {}
layer_counter = 0
total_values_dict = {}
zero_values_dict = {}
total_mac_ops = {}


class SparsityWeight(torch.nn.Module):
    def __init__(self):
        super(SparsityWeight, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(20, 128),
                        nn.ReLU(),
                        nn.Linear(128, 20),
                        nn.Sigmoid()
                    )

    def forward(self, x):
        out = self.model(x)
        return out


class Program(nn.Module):
    """
    Apply reprogramming.
    """
    def __init__(self, cfg, transform_path):
        super(Program, self).__init__()
        self.num_classes = 10
        self.cfg=cfg
        self.P = None
        self.init_transform(transform_path)

    def init_transform(self, transform_path):
        init_p = torch.ones((self.cfg.channels, self.cfg.h1, self.cfg.w1))
        # self.P = Parameter(init_p, requires_grad=True)
        self.P = Parameter((torch.rand(init_p.shape)) * 0.0001, requires_grad=True)
        # self.P = Parameter((torch.randn(init_p.shape) * 2 - 1) * 0.0001, requires_grad=True)
        if transform_path is not None:
            self.P = torch.load(transform_path, map_location=torch.device(device))['input_transform']

    def forward(self, image):
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        mean = mean[..., np.newaxis, np.newaxis]
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
        std = std[..., np.newaxis, np.newaxis]
        tmean = Parameter(torch.from_numpy(mean), requires_grad=False)
        tstd = Parameter(torch.from_numpy(std), requires_grad=False)
        tmean = tmean.to(device)
        tstd = tstd.to(device)
        x = image.data.clone()

        # x_adv = x + torch.tanh(self.P)
        # x_adv = (x_adv - tmean)/tstd
        # print(torch.min(x_adv), torch.max(x_adv))

        # x_adv = torch.tanh(x+self.P)
        # print(torch.min(x_adv), torch.max(x_adv))
        # x_adv = (x_adv - tmean)/tstd
        # print(torch.min(x_adv), torch.max(x_adv))

        # x_adv = torch.clamp(x+self.P, 0.0, 1.0)
        # x_adv = (x_adv - tmean) / tstd

        # pdb.set_trace()
        x_adv = 2 * x - 1
        x_adv = torch.tanh(0.5 * (torch.log(1 + x_adv + 1e-15) - torch.log(1 - x_adv + 1e-15)) + self.P)
        x_adv = 0.5 * x_adv + 0.5
        x_adv = (x_adv - tmean) / tstd

        # x_adv = x
        # x_adv = torch.tanh((torch.log(1 + x_adv + 1e-15) - torch.log(1 - x_adv + 1e-15)) + self.P)
        # print(torch.min(x_adv), torch.max(x_adv))
        # x_adv = (x_adv - tmean) / tstd

        return x_adv


class OutProgram(nn.Module):
    def __init__(self, cfg, transform_path):
        super(OutProgram, self).__init__()
        self.num_classes = 10
        self.cfg = cfg
        self.out_transform = torch.nn.Linear(self.num_classes, self.num_classes)
        # self.out_transform = nn.Sequential(
        #                         nn.Linear(10, 512),
        #                         nn.LeakyReLU(),
        #                         nn.Linear(512, 10),
        #                         nn.Sigmoid()
        #                     )
        # self.P_out = None
        # self.init_transform(transform_path)

    # def init_transform(self, transform_path):
    #     init_p = torch.ones((self.num_classes))
    #     self.P_out = Parameter((torch.randn(init_p.shape)) * 0.01, requires_grad=True)
    #     if transform_path is not None:
    #         self.P_out = torch.load(transform_path, map_location=torch.device(device))['input_transform']

    def forward(self, logits):
        logits_t = self.out_transform(logits)
        return logits_t


def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(
        labels.size(0)
    )  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds


def compute_sparsity_loss(lambda_loss):
    # pdb.set_trace()
    global layer_loss, layer_counter, total_values_dict, zero_values_dict, total_mac_ops
    activation_ = torch.nn.Tanh()
    beta = 100
    layer_wise_density = {}
    total_values_sum = sum(total_values_dict.values())
    zero_values_sum = sum(zero_values_dict.values())
    density = 1-zero_values_sum / total_values_sum
    layer_wise_density = [1.0-zero_values_dict[x]/total_values_dict[x] for x in zero_values_dict]

    total_macs = [total_mac_ops[x] for x in total_mac_ops]
    # l_sparsity = sum([torch.norm(x, p=2) for x in layer_loss.values()]) / total_values_sum
    # l_sparsity = sum([torch.sum(activation_(beta * x)) for x in layer_loss.values()])/total_values_sum
    # weighted loss per layer
    l_sparsity = 0
    # print(layer_loss)
    layer_loss_tensor = torch.empty(20)
    for i in range(20):
        layer_loss_tensor[i] = layer_loss[i]
    layer_loss_tensor = layer_loss_tensor.cuda()
    layer_loss_out = lambda_loss(layer_loss_tensor)
    # print(layer_loss_out)
    for lc in range(layer_loss_out.shape[0]):
        # l_sparsity += lambda_loss[lc]* layer_loss[lc]
        l_sparsity += layer_loss_out[lc]
        # layer_wise_density[layer_counter] = layer_loss[layer_counter]

    # l_sparsity = sum(( torch.sum(activation_(beta * layer_loss[x])) / total_values_dict[x] ) for x in layer_loss)
    return (l_sparsity, density, layer_wise_density, total_macs)


def comp_sparsity(self, input, output):
    global layer_loss, layer_counter, total_values_dict, zero_values_dict, total_mac_ops
	# if 'ReLU' in self.__class__.__name__ or 'MaxPool2d' in self.__class__.__name__:
    # if self.is_conv_input:
    #     o_shape = list(output.shape)
    #     total_values = o_shape[0] * o_shape[1] * o_shape[2] * o_shape[3]
    #     zero_values = torch.sum(output == 0)
    #     #total_values = reduce(mul, o_shape[1:], 1)
    #     #zero_values = torch.sum(output == 0, dim=[i for i in range(1, len(o_shape))]).to(dtype=torch.float)
    #     total_values_dict[layer_counter] = total_values
    #     zero_values_dict[layer_counter] = zero_values
    #     layer_loss[layer_counter] = output
    if "Conv2d" in self.__class__.__name__:
        # acts = input[0]
        # o_shape = list(acts.shape)
        # total_values = o_shape[0] * o_shape[1] * o_shape[2] * o_shape[3]
        # zero_values = torch.sum(acts == 0)
        # total_values_dict[layer_counter] = total_values
        # zero_values_dict[layer_counter] = zero_values
        # layer_loss[layer_counter] = acts

        # compute total MAC operations per image
        acts = input[0]
        a_shape = list(acts.shape)
        o_shape = list(output.shape)
        w_shape = list(self.weight.shape)
        total_macs =  a_shape[1] * w_shape[2] * w_shape[3] # per element of the output channel
        total_macs =  total_macs * o_shape[2] * o_shape[3] # per output channel
        total_macs = total_macs * o_shape[1] # all output channels ;
        total_macs = total_macs * o_shape[0]
        total_mac_ops[layer_counter] = total_macs

        # L2 norm based loss

        # group activations together and find density per group
        # Group tanh loss
        ind=0
        activation_ = torch.nn.Tanh()
        acts = input[0]
        beta = 100
        o_shape = list(acts.shape)
        total_values = o_shape[0] * o_shape[1] * o_shape[2] * o_shape[3]
        zero_values = torch.sum(acts == 0)
        total_values_dict[layer_counter] = total_values
        zero_values_dict[layer_counter] = zero_values

        # pdb.set_trace()
        # sum along channels
        group_density = torch.sum(activation_(beta*acts), 1) / o_shape[1]
        average_group_density_per_feature = torch.sum(group_density,[1,2]) / (o_shape[2]*o_shape[3])
        batch_group_density = torch.sum(average_group_density_per_feature)/o_shape[0]
        layer_loss[layer_counter] = batch_group_density

        layer_counter += 1
        # for b in range(o_shape[0]):
        #    for i in range(o_shape[2]):
        #        for j in range(o_shape[3]):
        #            group_acts = acts[b,:,i,j]
        #            l_sparsity += torch.sum(activation_(beta * group_acts))/o_shape[1]
        # layer_loss[layer_counter] = l_sparsity

        # # group lasso loss
        # acts = input[0]
        # grouped_act=torch.Tensor(i_shape[0]*i_shape[2]*i_shape[3], i_shape[1])
        # inx=0
        # for b in i_shape[0]:
        #     for i in i_shape[2]:
        #         for j in i_shape[3]:
        #             group_acts[inx,:] = acts[b,:,i,j]
        #             inx+=1


def init_hooks(arch, model):
    module_names = []
    hooks = {}
    for name, module in model.named_modules():
        module.module_name = name
        module.is_conv_input = False
        module_names.append(module.__class__.__name__)
        hooks[name] = module.register_forward_hook(comp_sparsity)
        module_idx = 0
    if arch == 'vgg16':
        for name, module in model.named_modules():
            if module_idx>=2 and module_idx < len(module_names)-1:
                module.is_conv_input = 'Conv2d' in module_names[module_idx+1] or 'Linear' in module_names[module_idx+1]
            module_idx += 1
    else:
        for name, module in model.named_modules():
            module.is_conv_input = 'ReLU' in module_names[module_idx]
            module_idx += 1


def transform_train(
    trainset,
    testset,
    checkpoint_path,
    num_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    lr_decay,
    lr_step,
    arch = "resnet18",
    dataset = "cifar10",
    in_channels = 3,
    precision = 8,
    force = False,
    fl = [],
    ber = 0.01,
    pos = -1,
    seed=0
):
    device = 'cuda'

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True

    (
        model,
        checkpoint_epoch
    ) = init_models(
        arch,
        in_channels,
        precision,
        True,
        checkpoint_path
    )

    # model = torch.nn.DataParallel(model)

    # stats.inspect_model(model)

    init_hooks(arch, model)
    Pg = Program(cfg, None)
    Opg = OutProgram(cfg, None)

    model, Pg, Opg = (
        model.to(device),
        Pg.to(device),
        Opg.to(device)
    )

    model.eval()

    sparsity_weight = SparsityWeight()
    sparsity_weight = sparsity_weight.to(device)

    sw_optimizer = torch.optim.SGD(
        sparsity_weight.parameters(),
        lr = 1e-3,
        weight_decay=weight_decay
    )

    sw_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        sw_optimizer,
        step_size=40,
        gamma=0.1
    )

    params = []
    for param in Pg.parameters():
        params.append(param)
    for param in Opg.parameters():
        params.append(param)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, params),
        lr=learning_rate,
        betas=(0.5, 0.999),
        weight_decay=weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_step,
        gamma=lr_decay
    )

    global layer_loss, layer_counter, total_values_dict, zero_values_dict

    for name, param in Pg.named_parameters():
        print("Param name: {}, grads is: {}".format(name, param.requires_grad))

    for name, param in Opg.named_parameters():
        print("Param name: {}, grads is: {}".format(name, param.requires_grad))

    print("========== Start training the parameter")

    for epoch in range(num_epochs):
        running_loss = 0
        running_sparsity_loss = 0
        running_correct = 0

        for batch_id, (image, label) in enumerate(trainloader):
            layer_counter = 0
            layer_loss = {}
            total_values_dict = {}
            zero_values_dict = {}

            # total_loss = 0
            image, label = image.to(device), label.to(device)
            # image_adv = image
            image_adv = Pg(image)
            out = model(image_adv)
            # logits_adv = out
            logits_adv = Opg(out)

            loss_orig, pred_orig = compute_loss(logits_adv, label)
            running_correct += torch.sum(pred_orig == label.data).item()

            sw_optimizer.zero_grad()
            # Pg.zero_grad()
            # Opg.zero_grad()
            optimizer.zero_grad()

            l_sparsity, _, _, _ = compute_sparsity_loss(sparsity_weight)
            running_sparsity_loss += l_sparsity.item()
            l_sparsity.backward(retain_graph = True)
            sw_optimizer.step()

            # total_loss = loss_orig + l_sparsity
            running_loss += loss_orig.item()
            loss_orig.backward()
            optimizer.step()

            # e_w_clone = []
            # with torch.no_grad():
            #     for param in params:
            #         numer = 0.05 * param.grad
            #         denom = torch.linalg.norm(param.grad, dim = 0, ord = 2)
            #         denom = torch.linalg.norm(denom, dim = 0, ord = 2)
            #         denom = torch.linalg.norm(denom, dim = 0, ord = 2)
            #         e_w = numer / denom
            #         e_w = e_w.cuda()
            #         e_w_clone.append(e_w)
            #         param += e_w

            # Pg.zero_grad()
            # Opg.zero_grad()

            # image_adv = Pg(image)
            # out = model(image_adv)
            # logits_adv = Opg(out)

            # loss_orig, pred_orig = compute_loss(logits_adv, label)

            # loss_orig.backward()

            # with torch.no_grad():
            #     i = 0
            #     for param in params:
            #         param -= learning_rate * param.grad
            #         param -= e_w_clone[i]
            #         i += 1

        accuracy_checking(model, trainloader, testloader, Pg, sparsity_weight, device, use_transform=True)
        lr_scheduler.step()
        sw_lr_scheduler.step()
        accuracy = running_correct / len(trainloader.dataset)
        running_loss = running_loss / len(trainloader)
        running_sparsity_loss = running_sparsity_loss / len(trainloader)
        print(
                "For epoch: {}, loss: {:.10f}, sparsity loss: {:.10f}, accuracy: {} ".format(
                epoch + 1,
                running_loss,
                running_sparsity_loss,
                accuracy
            )
        )

        if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs:
            torch.save(
                {
                    'input_transform': Pg.P,
                    'sparsity_weight': sparsity_weight.state_dict(),
                    'output_transform': Opg.state_dict()
                },
                '{}{}_W_{}.pt'.format(cfg.save_dir, arch, epoch + 1)
            )

    return running_loss, running_sparsity_loss, accuracy


def accuracy_checking(
    model, trainloader, testloader, pg, lambda_loss, device, use_transform=False
):
    total_train = 0
    total_test = 0
    correct_train = 0
    correct_test = 0
    global layer_loss, layer_counter, total_values_dict, zero_values_dict, total_mac_ops

    average_density = 0.0
    average_layer_wise_density = []
    for x, y in trainloader:
        layer_counter = 0
        layer_loss = {}
        total_values_dict = {}
        zero_values_dict = {}
        total_mac_ops = {}
        total_train += 1
        x, y = x.to(device), y.to(device)
        if use_transform:
            x_adv = pg(x)
            out = model(x_adv)
        else:
            out = model(x)
        _, pred = out.max(1)
        y = y.view(y.size(0))

        _, density, layer_wise_density,_  = compute_sparsity_loss(lambda_loss)
        # average_density += sum(density.cpu().numpy())
        average_density += (density.cpu().numpy())
        if average_layer_wise_density == []:
            average_layer_wise_density = [0.0 for x in layer_wise_density]
        for layer_num in range(len(average_layer_wise_density)):
            average_layer_wise_density[layer_num] += torch.sum(layer_wise_density[layer_num]).cpu().numpy()
        correct_train += torch.sum(pred == y.data).item()
    accuracy_train = correct_train / (len(trainloader.dataset))
    for layer_num in range(len(average_layer_wise_density)):
        average_layer_wise_density[layer_num] = average_layer_wise_density[layer_num] / len(trainloader)
    print(
            "Density of activations with training data {:.6f}".format(
        average_density/len(trainloader))
    )
    print(average_layer_wise_density)

    # logger = stats.DataLogger(
    #     int(len(testloader.dataset) / testloader.batch_size),
    #     testloader.batch_size,
    # )

    average_density = 0.0
    average_layer_wise_density = []
    average_layer_wise_macs = []
    for x, y in testloader:
        layer_counter = 0
        layer_loss = {}
        total_values_dict = {}
        zero_values_dict = {}
        total_mac_ops = {}

        total_test += 1
        x, y = x.to(device), y.to(device)
        if use_transform:
            x_adv = pg(x)
            out = model(x_adv)
        else:
            out = model(x)
        _, pred = out.max(1)
        y = y.view(y.size(0))
        _, density, layer_wise_density, macs = compute_sparsity_loss(lambda_loss)
        # average_density += sum(density.cpu().numpy())
        average_density += (density.cpu().numpy())

        # logger.update(out)

        if average_layer_wise_density == []:
            average_layer_wise_density = [0.0 for x in layer_wise_density]
            average_layer_wise_macs = [0 for x in layer_wise_density]
        for layer_num in range(len(average_layer_wise_density)):
            average_layer_wise_density[layer_num] += torch.sum(layer_wise_density[layer_num]).cpu().numpy()
            average_layer_wise_macs[layer_num] += (macs[layer_num])
        correct_test += torch.sum(pred == y.data).item()
    accuracy_test = correct_test / (len(testloader.dataset))
    for layer_num in range(len(average_layer_wise_density)):
        average_layer_wise_density[layer_num] = average_layer_wise_density[layer_num] / len(testloader)
        average_layer_wise_macs[layer_num] = average_layer_wise_macs[layer_num] / len(testloader)

    print(
            "Density of activations with testing data {:.6f}".format(
        average_density/len(testloader))
    )
    print(average_layer_wise_density)
    print("total mac ops")
    print(macs)
    # print("total mac ops with testing data %d"%(mac_ops/len(testloader)))

    print("Accuracy of training data {:5f}".format(accuracy_train))
    print("Accuracy of testing data {:5f}".format(accuracy_test))

    # logger.visualize()
    # stats.inspect_model(model)
    return accuracy_train, accuracy_test


if __name__ == "__main__":
    num_epochs = 50
    checkpoint_path = 'resnet18_cifar10_p_8_model_30_50_50.pth'

    training_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(90),
            transforms.ToTensor(),
        ]
    )
    testing_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=training_transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=testing_transform,
    )

    torch.autograd.set_detect_anomaly(True)

    transform_train(
        trainset,
        testset,
        checkpoint_path,
        num_epochs,
        32,     # batch size
        1e-2,   # learning rate
        5e-5,   # weight decay
        0.1,    # lr decay
        100,    # lr step
    )
