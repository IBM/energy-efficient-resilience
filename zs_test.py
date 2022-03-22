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

import zs_hooks_stats as stats
from config import cfg
from faultmodels import randomfault
from models import lenetf, resnetf, vggf

debug = False
visualize = False


def init_models(
    arch,
    bit_error_rate,
    precision,
    position,
    faulty_layers,
    checkpoint_path,
    device,
):

    in_channels = 3

    if not cfg.faulty_layers:
        """unperturbed model"""
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
    else:
        """Perturbed model, where the weights are injected with bit
        errors at the rate of ber"""
        rf = randomfault.RandomFaultModel(
            bit_error_rate, precision, position, 0
        )
        BitErrorMap0 = (
            torch.tensor(rf.BitErrorMap_flip0).to(torch.int32).to(device)
        )
        BitErrorMap1 = (
            torch.tensor(rf.BitErrorMap_flip1).to(torch.int32).to(device)
        )
        if arch == "vgg11":
            model = vggf(
                "A",
                in_channels,
                10,
                True,
                precision,
                bit_error_rate,
                position,
                BitErrorMap0,
                BitErrorMap1,
                faulty_layers,
            )
        elif arch == "vgg16":
            model = vggf(
                "D",
                in_channels,
                10,
                True,
                precision,
                bit_error_rate,
                position,
                BitErrorMap0,
                BitErrorMap1,
                faulty_layers,
            )
        elif arch == "resnet18":
            model = resnetf(
                "resnet18",
                10,
                precision,
                bit_error_rate,
                position,
                BitErrorMap0,
                BitErrorMap1,
                faulty_layers,
            )
        elif arch == "resnet34":
            model = resnetf(
                "resnet34",
                10,
                precision,
                bit_error_rate,
                position,
                BitErrorMap0,
                BitErrorMap1,
                faulty_layers,
            )
        else:
            model = lenetf(
                in_channels,
                10,
                precision,
                bit_error_rate,
                position,
                BitErrorMap0,
                BitErrorMap1,
                faulty_layers,
            )

    print(model)

    model = model.to(device)

    print("Restoring model from checkpoint", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("restored checkpoint at epoch - ", checkpoint["epoch"])
    print("Training loss =", checkpoint["loss"])
    print("Training accuracy =", checkpoint["accuracy"])
    # checkpoint_epoch = checkpoint["epoch"]

    return model


def inference(
    testloader,
    arch,
    dataset,
    ber,
    precision,
    position,
    checkpoint_path,
    faulty_layers,
    device,
):
    model = init_models(
        arch, ber, precision, -1, faulty_layers, checkpoint_path, device
    )
    if arch == "resnet18" or arch == "resnet34":
        stats.resnet_register_hooks(model, arch)
    if arch == "vgg16" or arch == "vgg11":
        stats.vgg_register_hooks(model, arch)
    logger = stats.DataLogger(
        int(len(testloader.dataset) / testloader.batch_size),
        testloader.batch_size,
    )

    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    print("Restoring model from checkpoint", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("restored checkpoint at epoch - ", checkpoint["epoch"])
    # print('Training loss =', checkpoint['loss'])
    print("Training accuracy =", checkpoint["accuracy"])

    model.eval()

    model = model.to(device)
    running_correct = 0.0

    with torch.no_grad():

        for t, (inputs, classes) in enumerate(testloader):

            inputs = inputs.to(device)
            classes = classes.to(device)
            model_outputs = model(inputs)
            # pdb.set_trace()
            lg, preds = torch.max(model_outputs, 1)
            correct = torch.sum(preds == classes.data)
            running_correct += correct

            logger.update(model_outputs)

    print(
        "Eval Accuracy %.3f"
        % (running_correct.double() / (len(testloader.dataset)))
    )

    # if arch=='resnet18' or arch=='resnet34':
    #    stats.inspect_model(model)
    #    stats.resnet_print_stats()
    # elif arch=='vgg16':
    #    stats.inspect_model(model)
    #    stats.vgg_print_stats()

    # logger.visualize()
