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

import re

import matplotlib.pyplot as plt
import numpy as np

# from gen_faultmap import *
# from defs import *
import torch
from torch import nn

# resnet_layer_1_nza = 0
# resnet_layer_1_nzw = 0
# resnet_layer_2_nza = 0
# resnet_layer_2_nzw = 0
# resnet_layer_3_nza = 0
# resnet_layer_3_nzw = 0
# resnet_layer_4_nza = 0
# resnet_layer_4_nzw = 0
# resnet_layer_5_nza = 0
# resnet_layer_5_nzw = 0

resnet_layer_1 = 0
resnet_layer_2 = 0
resnet_layer_3 = 0
resnet_layer_4 = 0
resnet_layer_5 = 0

vgg_layer_1 = 0
vgg_layer_2 = 0
vgg_layer_3 = 0
vgg_layer_4 = 0
vgg_layer_5 = 0
vgg_layer_6 = 0
vgg_layer_7 = 0
vgg_layer_8 = 0


debug = True


class DataLogger:
    def __init__(self, num_batches, batch_size):
        print(
            "init logger with number of batches %d, batch size %d"
            % (num_batches, batch_size)
        )
        self.confidences = np.zeros((num_batches, batch_size))
        self.confidence_diffs = np.zeros((num_batches, batch_size))
        self.predictions = np.zeros((num_batches, batch_size))
        self.logits = np.zeros((num_batches, batch_size))
        self.loss = np.zeros((num_batches))
        self.batch_id = 0
        self.batch_size = batch_size

    def update(self, model_outputs):
        softmax = nn.Softmax(dim=1)
        confidences = torch.max(softmax(model_outputs), 1)
        logits, predictions = torch.max(model_outputs, 1)
        self.confidences[self.batch_id, 0 : self.batch_size] = (
            confidences.values.detach().cpu().numpy()
        )
        self.predictions[self.batch_id, 0 : self.batch_size] = (
            predictions.detach().cpu().numpy()
        )
        self.logits[self.batch_id, 0 : self.batch_size] = (
            logits.detach().cpu().numpy()
        )
        self.batch_id += 1

    def visualize(self):
        print(np.histogram(self.confidences, 20))
        print(np.histogram(self.logits, np.arange(0, 60, 3)))

        # plot(self.confidences)

    # def writeLog(self,run,voltage,ber,err_type):
    #    if err_type == 1 or err_type == 3:
    #        h5file = arch + '_' + dataset + '_' + chip + '.h5'
    #        if os.path.exists(self.h5file):
    #            hf = h5py.File(h5file, 'a')
    #        else:
    #            hf = h5py.File(h5file, 'w')
    #        f_str = str(voltage)
    #        g = hf.create_group(f_str)
    #    else:
    #        h5file = arch + '_' + dataset + '_' + rand + '.h5'
    #        if os.path.exists(self.h5file):
    #            hf = h5py.File(h5file, 'a')
    #        else:
    #            hf = h5py.File(h5file, 'w')
    #        f_str = 'ber_'+str(ber) + '_run_' + str(run)_
    #        g = hf.create_group(f_str)

    #    g.create_dataset('confidences', data=self.confidences)
    #    g.create_dataset('confidence_diffs', data=self.confidence_diffs)
    #    g.create_dataset('predictions', data=self.predictions)
    #    g.create_dataset('logits', data=self.logits)


# end class


def inspect_model(model):
    for name, param in model.named_parameters():
        # inspect ranges for weights
        if re.match(".*weight$", name) is not None:
            print(param.size(), torch.min(param), torch.max(param))

    for t, (name, layer) in enumerate(model.named_modules()):
        print(name, layer)


def plot(data):
    # plt.subplot(len(faulty_layers),1,l+1)
    # lb = 'BER '+str(ber[v])
    # plt.hist(weights, bins=20, range=(-0.25,0.25), log=True, label=lb)
    plt.hist(data, bins=20, log=True)
    # plt.xlabel('')
    plt.ylabel("(log scale)")
    plt.legend()
    plt.grid(True)

    # if (visualize):
    #    faulty_layers = np.arange(0,72,3)
    #    params = list(model.parameters())
    #    for l,p in enumerate(params):
    #        weights=p.cpu().detach().numpy()
    #        if (np.isin(l,faulty_layers)):
    #            print(weights.shape)
    #            print('Max %.3f Min %.4f proportion of "
    #            "Non zero elements %.4f'%(np.max(weights),
    #            np.min(weights),
    #            np.count_nonzero(weights)/np.prod(weights.shape)))
    #        weights_r=np.reshape(weights_np, np.prod(weights_np.shape))
    # if l == 0:
    #            weights = weights_r
    # else:
    #            weights=np.concatenate((weights,weights_r), axis=0)
    #    plot(weights, v)


def resnet_layer_1_stats(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    # print('output size:', output.data.size())
    # osize = output.data.size().cpu().numpy()
    # non_zeros=torch.count_nonzero(output.data)
    # non_zeros=len(list(torch.nonzero(output.data).cpu()))
    global resnet_layer_1
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    resnet_layer_1 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def resnet_layer_2_stats(self, input, output):
    global resnet_layer_2
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    resnet_layer_2 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def resnet_layer_3_stats(self, input, output):
    global resnet_layer_3
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    resnet_layer_3 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def resnet_layer_4_stats(self, input, output):
    global resnet_layer_4
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    resnet_layer_4 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def resnet_layer_5_stats(self, input, output):
    global resnet_layer_5
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    resnet_layer_5 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def vgg_layer_1_stats(self, input, output):
    global vgg_layer_1
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    vgg_layer_1 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def vgg_layer_2_stats(self, input, output):
    global vgg_layer_2
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    vgg_layer_2 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def vgg_layer_3_stats(self, input, output):
    global vgg_layer_3
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    vgg_layer_3 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def vgg_layer_4_stats(self, input, output):
    global vgg_layer_4
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    vgg_layer_4 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def vgg_layer_5_stats(self, input, output):
    global vgg_layer_5
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    vgg_layer_5 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def vgg_layer_6_stats(self, input, output):
    global vgg_layer_6
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    vgg_layer_6 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def vgg_layer_7_stats(self, input, output):
    global vgg_layer_7
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    vgg_layer_7 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def vgg_layer_8_stats(self, input, output):
    global vgg_layer_8
    osize = torch.numel(output.data)
    non_zeros = (
        torch.numel(torch.nonzero(output.data)) / 4
    )  # torch.nonzero returns the non-zero element indices
    vgg_layer_8 += non_zeros / osize
    if debug:
        print("output non zero elements proportion %.3f" % (non_zeros / osize))


def resnet_register_hooks(model, arch):
    model.layer1[0].relu1.register_forward_hook(resnet_layer_1_stats)
    model.layer1[0].relu2.register_forward_hook(resnet_layer_1_stats)
    model.layer1[1].relu1.register_forward_hook(resnet_layer_1_stats)
    model.layer1[1].relu2.register_forward_hook(resnet_layer_1_stats)
    model.layer1[2].relu1.register_forward_hook(resnet_layer_1_stats)
    model.layer1[2].relu2.register_forward_hook(resnet_layer_1_stats)

    model.layer2[0].relu1.register_forward_hook(resnet_layer_2_stats)
    model.layer2[0].relu2.register_forward_hook(resnet_layer_2_stats)
    model.layer2[1].relu1.register_forward_hook(resnet_layer_2_stats)
    model.layer2[1].relu2.register_forward_hook(resnet_layer_2_stats)
    model.layer2[2].relu1.register_forward_hook(resnet_layer_2_stats)
    model.layer2[2].relu2.register_forward_hook(resnet_layer_2_stats)
    model.layer2[3].relu1.register_forward_hook(resnet_layer_2_stats)
    model.layer2[3].relu2.register_forward_hook(resnet_layer_2_stats)

    model.layer3[0].relu1.register_forward_hook(resnet_layer_3_stats)
    model.layer3[0].relu2.register_forward_hook(resnet_layer_3_stats)
    model.layer3[1].relu1.register_forward_hook(resnet_layer_3_stats)
    model.layer3[1].relu2.register_forward_hook(resnet_layer_3_stats)
    model.layer3[2].relu1.register_forward_hook(resnet_layer_3_stats)
    model.layer3[2].relu2.register_forward_hook(resnet_layer_3_stats)
    model.layer3[3].relu1.register_forward_hook(resnet_layer_3_stats)
    model.layer3[3].relu2.register_forward_hook(resnet_layer_3_stats)
    model.layer3[4].relu1.register_forward_hook(resnet_layer_3_stats)
    model.layer3[4].relu2.register_forward_hook(resnet_layer_3_stats)
    model.layer3[5].relu1.register_forward_hook(resnet_layer_3_stats)
    model.layer3[5].relu2.register_forward_hook(resnet_layer_3_stats)

    model.layer4[0].relu1.register_forward_hook(resnet_layer_4_stats)
    model.layer4[0].relu2.register_forward_hook(resnet_layer_4_stats)
    model.layer4[1].relu1.register_forward_hook(resnet_layer_4_stats)
    model.layer4[1].relu2.register_forward_hook(resnet_layer_4_stats)
    model.layer4[2].relu1.register_forward_hook(resnet_layer_4_stats)
    model.layer4[2].relu2.register_forward_hook(resnet_layer_4_stats)
    # elif arch == 'resnet34':
    #    model.layer1[0].relu1.register_forward_hook(resnet_layer_1_stats)
    #    model.layer2[0].relu1.register_forward_hook(resnet_layer_2_stats)
    #    model.layer3[0].relu1.register_forward_hook(resnet_layer_3_stats)
    #    model.layer4[0].relu1.register_forward_hook(resnet_layer_4_stats)


#        model.layer5[0].relu1.register_forward_hook(resnet_layer_5_stats)


def resnet_print_stats():
    print(
        "Proportion of non-zero elements in layer 1 %.3f"
        % (resnet_layer_1 / 600)
    )
    print(
        "Proportion of non-zero elements in layer 2 %.3f"
        % (resnet_layer_2 / 800)
    )
    print(
        "Proportion of non-zero elements in layer 3 %.3f"
        % (resnet_layer_3 / 1200)
    )
    print(
        "Proportion of non-zero elements in layer 4 %.3f"
        % (resnet_layer_4 / 600)
    )


# print('Proportion of non-zero elements "
# "in layer 5 %.3f'%(resnet_layer_5/100))


def vgg_register_hooks(model, arch):
    model.features[2].register_forward_hook(vgg_layer_1_stats)
    model.features[5].register_forward_hook(vgg_layer_2_stats)
    model.features[9].register_forward_hook(vgg_layer_3_stats)
    model.features[12].register_forward_hook(vgg_layer_4_stats)
    model.features[16].register_forward_hook(vgg_layer_5_stats)
    model.features[19].register_forward_hook(vgg_layer_6_stats)
    model.features[22].register_forward_hook(vgg_layer_7_stats)
    model.features[26].register_forward_hook(vgg_layer_8_stats)


#    model.features[29].register_forward_hook(vgg_layer_1_stats)
#    model.features[32].register_forward_hook(vgg_layer_2_stats)
#    model.features[36].register_forward_hook(vgg_layer_3_stats)
#    model.features[39].register_forward_hook(vgg_layer_4_stats)
#    model.features[42].register_forward_hook(vgg_layer_5_stats)


def vgg_print_stats():
    print("Proportion of non-zero elements %.3f" % (vgg_layer_1 / 100))
    print("Proportion of non-zero elements %.3f" % (vgg_layer_2 / 100))
    print("Proportion of non-zero elements %.3f" % (vgg_layer_3 / 100))
    print("Proportion of non-zero elements %.3f" % (vgg_layer_4 / 100))
    print("Proportion of non-zero elements %.3f" % (vgg_layer_5 / 100))
    print("Proportion of non-zero elements %.3f" % (vgg_layer_6 / 100))
    print("Proportion of non-zero elements %.3f" % (vgg_layer_7 / 100))
    print("Proportion of non-zero elements %.3f" % (vgg_layer_8 / 100))
