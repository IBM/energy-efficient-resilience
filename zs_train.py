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

import os
import sys

import torch
import torch.optim as optim 
from torch import nn

from config import cfg
from models import default_model_path, init_models_faulty, init_models

import torch.nn.utils.prune as prune
from quantized_ops import zs_quantized_ops

__all__ = ["training"]

debug = False
torch.manual_seed(0)

class WarmUpLR(optim.lr_scheduler._LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def training(
    trainloader,
    arch,
    dataset,
    in_channels,
    precision,
    retrain,
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
    :param arch: A string. The architecture of the model would be used.
    :param dataset: A string. The name of the training data.
    :param in_channels: An int. The input channels of the training data.
    :param precision: An int. The number of bits would be used to quantize
                              the model.
    :param retrain: A boolean. Start from checkpoint.
    :param checkpoint_path: A string. The path that stores the models.
    :param force: Overwrite checkpoint.
    :param device: A string. Specify using GPU or CPU.
    """

    model, checkpoint_epoch = init_models(arch, 3, precision, retrain, checkpoint_path, dataset) # Quantization Aware Training without using bit error!

    print("Training with Learning rate %.4f" % (cfg.learning_rate))

    if dataset == 'cifar100': 
        print('cifar100')
        opt = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
        #iter_per_epoch = len(trainloader)
        #warmup_scheduler = WarmUpLR(opt, iter_per_epoch * 1) # warmup = 1
        train_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=cfg.lr_step, gamma=cfg.lr_decay)

    else:
        opt = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
        opt, milestones=cfg.lr_step, gamma=cfg.lr_decay
    )

    model = model.to(device)
    #from torchsummary import summary
    #if dataset == 'tinyimagenet':
    #    summary(model, (3, 64, 64))
    #else:
    #    summary(model, (3, 32, 32))
    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    
    for name, module in model.named_modules():
        if isinstance(module, zs_quantized_ops.nnConv2dSymQuant):
            prune.l1_unstructured(module, name='weight', amount=0.5)
        elif isinstance(module, zs_quantized_ops.nnLinearSymQuant):
            prune.l1_unstructured(module, name='weight', amount=0.5)

    for x in range(checkpoint_epoch + 1, cfg.epochs):

        print("Epoch: %03d" % x)

        running_loss = 0.0
        running_correct = 0
        for batch_id, (inputs, outputs) in enumerate(trainloader):
            
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            opt.zero_grad()

            model.train()
            model_outputs = model(inputs)  # pylint: disable=E1102

            _, preds = torch.max(model_outputs, 1)
            outputs = outputs.view(
                outputs.size(0)
            )  # changing the size from (batch_size,1) to batch_size.

            loss = nn.CrossEntropyLoss()(model_outputs, outputs)

            # Compute gradient of perturbed weights with perturbed loss
            loss.backward()

            # update restored weights with gradient
            opt.step()
            if dataset == 'cifar100': 
                train_scheduler.step()
            #    if x <= 1: # warmup = 1
            #        warmup_scheduler.step()
            #    else:
            #        train_scheduler.step()
            # lr_scheduler.step()

            running_loss += loss.item()
            running_correct += torch.sum(preds == outputs.data)

        if dataset == 'cifar10':
            lr_scheduler.step()

        accuracy = running_correct.double() / (len(trainloader.dataset))
        print(
            "For epoch: {}, loss: {:.6f}, accuracy: {:.5f}".format(
                x, running_loss / len(trainloader.dataset), accuracy
            )
        )
        if (x+1)%10 == 0:

            model_path = default_model_path(
                cfg.model_dir, arch, dataset, precision, fl, ber, pos, x+1
            )

            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))

            if os.path.exists(model_path) and not force:
                print("Checkpoint already present ('%s')" % model_path)
                sys.exit(1)
                
            # save pruned weights
            state_dict = model.state_dict().copy()
            weight_orig = torch.tensor(0)
            for key, val in model.state_dict().items():
                if 'weight_orig' in key:
                    weight_orig = val
                if 'weight_mask' in key:
                    module_name = '.'.join(key.split('.')[:-1])
                    del state_dict[module_name + '.weight_orig']
                    del state_dict[module_name + '.weight_mask']
                    state_dict[module_name + '.weight'] = torch.mul(weight_orig, val)
                    

            torch.save(
                {
                    "epoch": x,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": running_loss / batch_id,
                    "accuracy": accuracy,
                },
                model_path,
            )
