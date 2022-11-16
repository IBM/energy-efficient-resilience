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

#from vgg import VGG

import zs_hooks_stats as stats
from operator import mul
from functools import reduce

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

layer_loss = {}
layer_counter = 0
total_values_dict = {}
zero_values_dict = {}
total_mac_ops = {}
import pdb
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

    # Initialize Perturbation
    def init_transform(self, transform_path):
        init_p = torch.ones((self.cfg.channels, self.cfg.h1, self.cfg.w1))
        #self.P = Parameter(init_p, requires_grad=True)
        self.P = Parameter((torch.randn(init_p.shape) * 2 - 1) * 0.0001, requires_grad=True)
        #self.P = Parameter(torch.randn(init_p.shape), requires_grad=True)
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
        #print(torch.min(x), torch.max(x))
        #y = (x - tmean)/tstd
        #print(torch.min(y), torch.max(y))
        #pdb.set_trace()
        #x_adv = x + self.tanh_fn(self.P)
        #x_adv = torch.clamp(x_adv, min=-1, max=1)


        #x_adv = torch.tanh(x+self.P)
        #print(torch.min(x_adv), torch.max(x_adv))
        #x_adv = (x_adv - tmean)/tstd
        #print(torch.min(x_adv), torch.max(x_adv))
        
        #x_adv = torch.clamp(x+self.P, 0.0, 1.0)
        #x_adv = (x_adv - tmean) / tstd

        #pdb.set_trace()
        #x_adv = 2 * x - 1
        #x_adv = torch.tanh(0.5 * (torch.log(1 + x_adv + 1e-15) - torch.log(1 - x_adv + 1e-15)) + self.P)
        #x_adv = 0.5 * x_adv + 0.5
        #x_adv = (x_adv - tmean) / tstd
        x_adv = x
        x_adv = torch.tanh((torch.log(1 + x_adv + 1e-15) - torch.log(1 - x_adv + 1e-15)) + self.P)
        #x_adv = (x_adv - tmean) / tstd
        return x_adv


def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(
        labels.size(0)
    )  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds



def compute_sparsity_loss(lambda_loss):
    #pdb.set_trace()
    global layer_loss, layer_counter, total_values_dict, zero_values_dict, total_mac_ops
    activation_ = torch.nn.Tanh()
    beta = 100
    layer_wise_density = {}
    total_values_sum = sum(total_values_dict.values())
    zero_values_sum = sum(zero_values_dict.values())
    density = 1-zero_values_sum / total_values_sum
    layer_wise_density = [1.0-zero_values_dict[x]/total_values_dict[x] for x in zero_values_dict]

    total_macs = [total_mac_ops[x] for x in total_mac_ops]
#    l_sparsity = sum([torch.norm(x, p=2) for x in layer_loss.values()]) / total_values_sum
    #l_sparsity = sum([torch.sum(activation_(beta * x)) for x in layer_loss.values()])/total_values_sum
    #weighted loss per layer 
    l_sparsity = 0
    for layer_counter in layer_loss:
        l_sparsity += lambda_loss[layer_counter]* layer_loss[layer_counter]
        #layer_wise_density[layer_counter] = layer_loss[layer_counter]

    #l_sparsity = sum(( torch.sum(activation_(beta * layer_loss[x])) / total_values_dict[x] ) for x in layer_loss)
    return (l_sparsity, density, layer_wise_density, total_macs)




def comp_sparsity(self, input, output):
    global layer_loss, layer_counter, total_values_dict, zero_values_dict, total_mac_ops
	#if 'ReLU' in self.__class__.__name__ or 'MaxPool2d' in self.__class__.__name__:
#    if self.is_conv_input:
#        o_shape = list(output.shape)
#        total_values = o_shape[0] * o_shape[1] * o_shape[2] * o_shape[3]
#        zero_values = torch.sum(output == 0)
#        #total_values = reduce(mul, o_shape[1:], 1)
#        #zero_values = torch.sum(output == 0, dim=[i for i in range(1, len(o_shape))]).to(dtype=torch.float)
#        total_values_dict[layer_counter] = total_values
#        zero_values_dict[layer_counter] = zero_values
#        layer_loss[layer_counter] = output

    if "Conv2d" in self.__class__.__name__:

        if layer_counter < 50:
            #acts = input[0]
            #o_shape = list(acts.shape)
            #total_values = o_shape[0] * o_shape[1] * o_shape[2] * o_shape[3]
            #zero_values = torch.sum(acts == 0)
            #total_values_dict[layer_counter] = total_values
            #zero_values_dict[layer_counter] = zero_values
            #layer_loss[layer_counter] = acts


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
            
            #pdb.set_trace()
            # sum along channels 
            group_density = torch.sum(activation_(beta*acts), 1) / o_shape[1] 
            average_group_density_per_feature = torch.sum(group_density,[1,2]) / (o_shape[2]*o_shape[3])
            batch_group_density = torch.sum(average_group_density_per_feature)/o_shape[0]
            layer_loss[layer_counter] = batch_group_density

            
            layer_counter += 1
        #for b in range(o_shape[0]):
        #    for i in range(o_shape[2]):
        #        for j in range(o_shape[3]):
        #            group_acts = acts[b,:,i,j]
        #            l_sparsity += torch.sum(activation_(beta * group_acts))/o_shape[1]
        #layer_loss[layer_counter] = l_sparsity            
#
#       # group lasso loss 
#        acts = input[0]
#        grouped_act=torch.Tensor(i_shape[0]*i_shape[2]*i_shape[3], i_shape[1])
#        inx=0
#        for b in i_shape[0]:
#            for i in i_shape[2]:
#                for j in i_shape[3]:
#                    group_acts[inx,:] = acts[b,:,i,j]
#                    inx+=1
#
#
#
#                
#
#
#

def init_hooks(arch,model):

    module_names = []
    hooks = {}
    for name, module in model.named_modules():
        module.module_name = name
        module.is_conv_input = False
        module_names.append(module.__class__.__name__)
        hooks[name] = module.register_forward_hook(comp_sparsity)

        #print(module_names)
        module_idx = 0
    if arch == 'vgg16':
        for name, module in model.named_modules():
            if module_idx>=2 and module_idx < len(module_names)-1:
                module.is_conv_input = 'Conv2d' in module_names[module_idx+1] or 'Linear' in module_names[module_idx+1]
                #module.is_conv_input = 'Conv2d' in module_names[module_idx+1] or 'Linear' in module_names[module_idx+1]
            module_idx += 1
    else:
        for name, module in model.named_modules():
            module.is_conv_input = 'ReLU' in module_names[module_idx]
            module_idx += 1

def check_sparsity_accuracy(
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

    #model = VGG('VGG16') 
    #checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    #model.load_state_dict(checkpoint['net'])


    #assert checkpoint_epoch == checkpoint_epoch_perturbed

    init_hooks(arch, model)
    Pg = Program(cfg, None)
    model, Pg = (
        model.to(device),
        Pg.to(device)
    )

    model.eval()

    accuracy_checking(model, trainloader, testloader, Pg, device, use_transform=True)



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
        checkpoint_epoch
    ) = init_models(
        arch,
        in_channels,
        precision,
        True,
        checkpoint_path
    )

    #model = VGG('VGG16') 
    #checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    #model.load_state_dict(checkpoint['net'])


    #assert checkpoint_epoch == checkpoint_epoch_perturbed
    stats.inspect_model(model)
    init_hooks(arch, model)
    Pg = Program(cfg, None)
    model, Pg = (
        model.to(device),
        Pg.to(device)
    )

    model.eval()

    #accuracy_checking(model, trainloader, testloader, Pg, device, use_transform=False)
    accuracy_checking(model, trainloader, testloader, Pg, cfg.lb, device, use_transform=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, Pg.parameters()),
        lr=cfg.learning_rate,
        betas=(0.5, 0.999),
        weight_decay=cfg.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.lr_step, gamma=cfg.lr_decay
    )
    global layer_loss, layer_counter, total_values_dict, zero_values_dict

    for name, param in Pg.named_parameters():
        print("Param name: {}, grads is: {}".format(name, param.requires_grad))

    print(
        "========== Start training the parameter"
    
    )
    for epoch in range(cfg.epochs):
        running_loss = 0
        running_sparsity_loss = 0
        running_correct = 0
    
              
        for batch_id, (image, label) in enumerate(trainloader):

            layer_counter = 0
            layer_loss = {}
            total_values_dict = {}
            zero_values_dict = {} 

            total_loss = 0  
            image, label = image.to(device), label.to(device)
            image_adv = Pg(image)  
            out = model(image_adv) 
            loss_orig, pred_orig = compute_loss(out, label)
            running_correct += torch.sum(pred_orig == label.data).item()
            l_sparsity, density, layerwise_density,_ = compute_sparsity_loss(cfg.lb)
            total_loss = loss_orig + l_sparsity
            #total_loss = l_sparsity
            running_loss += total_loss.item()
            running_sparsity_loss += l_sparsity.item()
            optimizer.zero_grad()
            # Pg.zero_grad()
            total_loss.backward()
            optimizer.step()
            #print(Pg.P.grad)
            
            
            

        lr_scheduler.step()
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
        #if ((epoch+1) % 5 == 0):
        #    accuracy_checking(model, trainloader, testloader, Pg, device, use_transform=True)
        if (epoch + 1) % 20 == 0 or (epoch + 1) == cfg.epochs:
            torch.save({'input_transform': Pg.P},'{}/{}_W_{}.pt'.format(cfg.save_dir, arch, epoch + 1))
            accuracy_checking(model, trainloader, testloader, Pg, cfg.lb, device, use_transform=True)

            #lb += 0.5
            #print("Lambda value: ", lb)



def accuracy_checking(
    model, trainloader, testloader, pg, lambda_loss, device, use_transform=False
):
    # For training data first:
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
        #average_density += sum(density.cpu().numpy())
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
    
    
    
    logger = stats.DataLogger(
        int(len(testloader.dataset) / testloader.batch_size),
        testloader.batch_size,
    )
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
        #average_density += sum(density.cpu().numpy())
        average_density += (density.cpu().numpy())
            
        logger.update(out)

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
    #print("total mac ops with testing data %d"%(mac_ops/len(testloader)))
    


    print("Accuracy of training data {:5f}".format(accuracy_train))
    print("Accuracy of testing data {:5f}".format(accuracy_test))
   
    #logger.visualize()
    #stats.inspect_model(model)
    return accuracy_train, accuracy_test




def transform_test(
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

    #assert checkpoint_epoch == checkpoint_epoch_perturbed

    init_hooks(arch, model)

    model.eval()


    transform_path = cfg.save_dir + '/resnet18_W_1.pt'
    Pg = Program(cfg, transform_path)
    Pg = Pg.to(device)
    model, Pg = (
        model.to(device),
        Pg.to(device)
    )
  
    # Setting without input transformation
    accuracy_train_list = []
    accuracy_test_list = []

    # Setting with input transformation
    accuracy_train_list_with_transformation = []
    accuracy_test_list_with_transformation = []

    # Without using transform
    accuracy_train, accuracy_test = accuracy_checking(model, trainloader, testloader, Pg, device, use_transform=False)
    accuracy_train_list.append(accuracy_train)
    accuracy_test_list.append(accuracy_test)

    # With input transform
    accuracy_train, accuracy_test = accuracy_checking(model, trainloader, testloader, Pg, device, use_transform=True)
    accuracy_train_list_with_transformation.append(accuracy_train)
    accuracy_test_list_with_transformation.append(accuracy_test)

    # Without using transform
    print('The average results without input transformation -> accuracy_train: {:5f}, accuracy_test: {:5f},' .format(
      np.mean(accuracy_train_list), np.mean(accuracy_test_list)
    ))
    print('The average results without input transformation -> std_accuracy_train: {:5f}, std_accuracy_test: {:5f}'.format(
      np.std(accuracy_train_list), np.std(accuracy_test_list)
    ))

    print()

    # With input transform
    print('The average results with input transformation -> accuracy_train: {:5f}, accuracy_test: {:5f} '.format(
      np.mean(accuracy_train_list_with_transformation), np.mean(accuracy_test_list_with_transformation), 
    ))
    print('The average results with input transformation -> std_accuracy_train: {:5f}, std_accuracy_test: {:5f}'.format(
      np.std(accuracy_train_list_with_transformation), np.std(accuracy_test_list_with_transformation)
    ))
