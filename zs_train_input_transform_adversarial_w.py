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
import copy
import os

from config import cfg
from models import init_models_pairs


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-20


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


def draw_learning_curve(loss_list, epoch, arch, ber, lb):
    epoch_list = [e + 1 for e in range(epoch)]
    plt.plot(epoch_list, loss_list)
    plt.title("Learning curve of the parameter P")
    plt.savefig(
        cfg.save_dir_curve
        + "adversarial_arch_{}_LR_a{}_p{}_E_{}_PGD_{}_ber_{}_lb_{}.png".format(
            arch,
            cfg.learning_rate_adversarial,
            cfg.learning_rate,
            cfg.epochs,
            cfg.PGD_STEP,
            ber,
            lb,
        )
    )


def check_dir(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(
        labels.size(0)
    )  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds


def accuracy_checking(
    model_orig,
    model_p,
    trainloader,
    testloader,
    pg,
    device,
    use_transform=False,
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

    if use_transform:
        print("----- With input transformation: -----")
    else:
        print("----- Without using input transformation: -----")
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

    return (
        accuracy_orig_train,
        accuracy_p_train,
        accuracy_orig_test,
        accuracy_p_test,
    )


def diff_in_weight(model_orig, model_p):
    diff_dict = OrderedDict()
    for (old_k, old_w), (new_k, new_w) in zip(
        model_orig.named_parameters(), model_p.named_parameters()
    ):
        if old_w.requires_grad:
            diff_w = new_w - old_w
            diff_dict[old_k] = diff_w
    return diff_dict


def add_into_weights(model_p, diff):
    names_diff = diff.keys()
    for name, param in model_p.named_parameters():
        if name in names_diff:
            param.data = param.data + diff[name]


def quantization(model_weights, precision=8):
    max_val = torch.max(torch.abs(model_weights))
    delta = max_val / (2 ** (precision - 1) - 1)
    return delta


def pgd(model_orig, model_p, epsilon, alpha):
    for (name_orig, param_orig), (name_p, param_p) in zip(
        model_orig.named_parameters(), model_p.named_parameters()
    ):
        if param_p.requires_grad == True:
            delta = quantization(param_p)
            param_p.data = param_p.data + (alpha * param_p.grad.sign())
            param_p.data = torch.clamp(
                param_p.data,
                min=param_orig.data - (epsilon * delta),
                max=param_orig.data + (epsilon * delta),
            )


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

    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    Pg = Program(cfg)

    lb = cfg.lb  # Lambda

    print(
        "========== Check setting: Epoch: {}, Batch_size: {}, Lambda: {}, BitErrorRate: {} ==========".format(
            cfg.epochs, cfg.batch_size, lb, ber
        )
    )
    print(
        "========== Check PGD setting: STEP: {}, LR_p: {}, LR_a: {} ==========".format(
            cfg.PGD_STEP, cfg.learning_rate, cfg.alpha
        )
    )
    print(
        "========== Check folder: save_dir: {}, save_dir_curve: {} ==========".format(
            cfg.save_dir, cfg.save_dir_curve
        )
    )

    print(
        "========== Start training the parameter of the input transform by using Adversarial Training =========="
    )

    (model, checkpoint_epoch, _, _) = init_models_pairs(
        arch,
        in_channels,
        precision,
        True,
        checkpoint_path,
        fl,
        ber,
        pos,
        seed=0,
    )  # Create clean model.

    EPSILON = ber * 511

    learning_curve_loss = []

    # Using Adam:
    # optimizer = torch.optim.Adam(
    #    filter(lambda p: p.requires_grad, Pg.parameters()),
    #    lr=cfg.learning_rate,
    #    betas=(0.5, 0.999),
    #    weight_decay=5e-4,
    # )

    # Using SGD:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=cfg.decay
    )

    for epoch in range(cfg.epochs):
        running_loss = 0
        running_correct_orig = 0
        running_correct_p = 0

        # For each epoch, we will use N perturbed model for training.
        for batch_id, (image, label) in enumerate(trainloader):

            model_perturbed = copy.deepcopy(
                model
            )  # Because we can't modify the origin model weights, therefore, copy a new model to apply ADV attack.
            model.eval()
            model_perturbed.train()

            # Apply AWP with PGD!
            for i in range(cfg.PGD_STEP):
                loss_pgd = 0
                model, model_perturbed = model.to(device), model_perturbed.to(
                    device
                )
                for param in model_perturbed.parameters():
                    param.requires_grad = True
                image, label = image.to(device), label.to(device)
                out_perturb = model_perturbed(image)
                loss_pgd, pred_pgd = compute_loss(out_perturb, label)
                loss_pgd.backward()
                pgd(model, model_perturbed, EPSILON, cfg.alpha)
                # print("loss_pgd: {}".format(loss_pgd.item()))
                # for param_a, param_p in zip(model.parameters(), model_perturbed.parameters()):
                #    print(param_p - param_a)

            diff_w = diff_in_weight(
                model, model_perturbed
            )  # Calculate the different between original model and perturbed model.

            # Start to update original model weights that try to let the model to be robust.
            loss = 0
            image, label = image.to(device), label.to(device)

            model_perturbed = copy.deepcopy(model)
            add_into_weights(model_perturbed, diff_w)
            model, model_perturbed = model.to(device), model_perturbed.to(
                device
            )
            model.train()
            model_perturbed.train()
            out, out_perturb = model(image), model_perturbed(image)

            # Compute the loss for clean model and perturbed model
            loss_orig, pred_orig = compute_loss(out, label)
            loss_perturb, pred_perturb = compute_loss(out_perturb, label)

            # Keep the running accuracy of clean model and perturbed model.
            running_correct_orig += torch.sum(pred_orig == label.data).item()
            running_correct_p += torch.sum(pred_perturb == label.data).item()

            # print("-- loss_orig --: {}".format(loss_orig.item()))
            # print("-- loss_perturb --: {}".format(loss_perturb.item()))
            running_loss += loss_orig.item() + loss_perturb.item()

            optimizer.zero_grad()

            # Calculate gradients
            # We can't calculate ∂loss_perturb/∂w (Because no data flow into clean model).
            # So, first, calculate the gradient ∂loss_perturb/∂w':
            for param in model_perturbed.parameters():
                param.requires_grad = True
            loss_perturb.backward()

            # Second, calculate the gradient loss_orig/∂w:
            loss_orig.backward()

            # Add gradients from perturbed model and clean model:
            for param_orig, param_p in zip(
                model.parameters(), model_perturbed.parameters()
            ):
                if param_orig.requires_grad:
                    param_orig.grad += param_p.grad

            # Update based on the gradients
            optimizer.step()
            lr_scheduler.step()

        learning_curve_loss.append(running_loss)
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

    # draw_learning_curve(learning_curve_loss, cfg.epochs, arch, ber, lb)

    # -------------------------------------------------- Inference --------------------------------------------------

    print(
        "========== Start checking the accuracy with different perturbed model: Random bit error mode =========="
    )

    # model: Adversarial training roubstness model.
    # model_orig_clean: Original clean model.
    # model_perturbed: Adversarial training model.
    # model_rbe: Model with bit error made from origin clean model.
    # model_rbe_robust: Model with bit error made from robust model.
    for i in range(50000, 50010):

        print(" ********** For seed: {} ********** ".format(i))

        # We load the model with random bit error(model_rbe), later we will load the weights trained by ADV training into model_rbe.
        (_, _, model_rbe, _) = init_models_pairs(
            arch,
            in_channels,
            precision,
            True,
            checkpoint_path,
            fl,
            ber,
            pos,
            seed=i,
        )

        # loading robust model weights into model_rbe_robust here.
        model_rbe_robust = copy.deepcopy(model_rbe)
        model_rbe_robust.load_state_dict(model.state_dict())

        model, model_rbe, model_rbe_robust = (
            model.to(device),
            model_rbe.to(device),
            model_rbe_robust.to(device),
        )
        model.eval()
        model_rbe.eval()
        model_rbe_robust.eval()

        # Roubstness model with bit error
        print("Testing origin bit error model: ")
        (
            accuracy_orig_train,
            accuracy_p_train,
            accuracy_orig_test,
            accuracy_p_test,
        ) = accuracy_checking(
            model,
            model_rbe,
            trainloader,
            testloader,
            Pg,
            device,
            use_transform=False,
        )

        # Roubstness model with bit error
        print("Testing robust bit error model: ")
        (
            accuracy_orig_train,
            accuracy_p_train,
            accuracy_orig_test,
            accuracy_p_test,
        ) = accuracy_checking(
            model,
            model_rbe_robust,
            trainloader,
            testloader,
            Pg,
            device,
            use_transform=False,
        )
