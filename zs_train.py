#!/usr/bin/env python
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

from config import cfg
from models import resnetf
from models import vggf
from models import lenetf
import sys
import torch

from torch import nn
import torch.optim as optim

# import torch.nn.functional as F
# import torchvision

debug = False

sys.path.append("./models")
torch.manual_seed(0)


def init_models(arch, precision, retrain, checkpoint_path):

    in_channels = 3

    """ unperturbed model
    """
    if arch == "vgg11":
        model = vggf("A", in_channels, 10, True, precision, 0, 0, 0, 0, [])
    elif arch == "vgg16":
        model = vggf("D", in_channels, 10, True, precision, 0, 0, 0, 0, [])
    elif arch == "resnet18":
        model = resnetf("resnet18", 10, precision, 0, 0, 0, 0, [])
    elif arch == "resnet34":
        model = resnetf("resnet34", 10, precision, 0, 0, 0, 0, [])
    else:
        model = lenetf(in_channels, 10, precision, 0, 0, 0, 0, [])

    print(model)

    checkpoint_epoch = 0
    if retrain:
        print("Restoring model from checkpoint", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        print("restored checkpoint at epoch - ", checkpoint["epoch"])
        print("Training loss =", checkpoint["loss"])
        print("Training accuracy =", checkpoint["accuracy"])
        checkpoint_epoch = checkpoint["epoch"]

    return model, checkpoint_epoch


def training(
    trainloader, arch, dataset, precision, retrain, checkpoint_path, device
):

    model, checkpoint_epoch = init_models(
        arch, precision, retrain, checkpoint_path
    )

    print("Training with Learning rate %.4f" % (cfg.learning_rate))
    opt = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)

    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    curr_lr = cfg.learning_rate
    for x in range(cfg.epochs):

        running_loss = 0.0
        running_correct = 0
        for batch_id, (inputs, outputs) in enumerate(trainloader):
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            opt.zero_grad()

            # Store original model parameters before
            # quantization/perturbation, detached from graph
            if precision > 0:
                list_init_params = []
                with torch.no_grad():
                    for init_params in model.parameters():
                        list_init_params.append(init_params.clone().detach())

                if debug:
                    if batch_id % 100 == 0:
                        print("initial params")
                        print(model.fc2.weight[0:3, 0:3])
                        print(model.conv1.weight[0, 0, :, :])

            model_outputs = model(inputs)  # pylint: disable=E1102

            _, preds = torch.max(model_outputs, 1)
            outputs = outputs.view(
                outputs.size(0)
            )  # changing the size from (batch_size,1) to batch_size.

            if precision > 0:
                if debug:
                    if batch_id % 100 == 0:
                        print("quantized params")
                        print(model.fc2.weight[0:3, 0:3])
                        print(model.conv1.weight[0, 0, :, :])

            loss = nn.CrossEntropyLoss()(model_outputs, outputs)

            if debug:
                print("epoch %d batch %d loss %.3f" % (x, batch_id, loss))

            # Compute gradient of perturbed weights with perturbed loss
            loss.backward()

            # restore model weights with unquantized value
            if precision > 0:
                with torch.no_grad():
                    for i, restored_params in enumerate(model.parameters()):
                        restored_params.copy_(list_init_params[i])

                if debug:
                    if batch_id % 100 == 0:
                        print("restored params")
                        print(model.fc2.weight[0:3, 0:3])
                        print(model.conv1.weight[0, 0, :, :])

            # update restored weights with gradient
            opt.step()

            running_loss += loss.item()
            running_correct += torch.sum(preds == outputs.data)

        # update learning rate
        if (x == 80) or (x == 120):
            curr_lr /= 10.0
            for param_group in opt.param_groups:
                param_group["lr"] = curr_lr
            print("Training with Learning rate %.4f" % (curr_lr))

        accuracy = running_correct.double() / (len(trainloader.dataset))
        print(
            "epoch %d loss %.6f accuracy %.6f"
            % (x, running_loss / (batch_id), accuracy)
        )
        # writer.add_scalar('Loss/train', running_loss/batch_id, x)
        # ## loss/#batches
        if ((x) % 40 == 0) or (x == cfg.epochs - 1):
            model_path = (
                "/tmp/"
                + arch
                + "_"
                + dataset
                + "_p_"
                + str(precision)
                + "_model_"
                + str(checkpoint_epoch + x)
                + ".pth"
            )
            torch.save(
                {
                    "epoch": (checkpoint_epoch + x),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": running_loss / batch_id,
                    "accuracy": accuracy,
                },
                model_path,
            )
            # utils.collect_gradients(params, faulty_layers)
