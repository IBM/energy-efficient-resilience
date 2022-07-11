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

import torch

from config import cfg
from faultmodels import randomfault
import faultsMap as fmap

from .lenet import lenet  # noqa: F401
from .lenetf import lenetf  # noqa: F401
from .resnet import resnet  # noqa: F401
from .resnetf import resnetf  # noqa: F401
from .simplenet import simplenet  # noqa: F401
from .vgg import vgg  # noqa: F401
from .vggf import vggf  # noqa: F401

# Create the fault map from randomfault module.
def create_faults(
    precision,
    bit_error_rate,
    position,
    seed=0,
):
    rf = randomfault.RandomFaultModel(
        bit_error_rate, precision, position, seed=seed
    )
    fmap.BitErrorMap0 = (
        torch.tensor(rf.BitErrorMap_flip0).to(torch.int32).to(cfg.device)
    )
    fmap.BitErrorMap1 = (
        torch.tensor(rf.BitErrorMap_flip1).to(torch.int32).to(cfg.device)
    )


def init_models(arch, in_channels, precision, retrain, checkpoint_path, dataset='cifar10'):

    """
    Default model loader
    """
    classes = 10
    if dataset == 'cifar10':
        classes = 10
    elif dataset == 'cifar100':
        classes = 100
    elif dataset == 'tinyimagenet':
        classes = 200
    elif dataset == 'gtsrb':
        classes = 43
    else:
        classes = 10


    if arch == "vgg11":
        model = vggf("A", in_channels, classes, True, precision, 0, 0, [])
    elif arch == "vgg16":
        model = vggf("D", in_channels, classes, True, precision, 0, 0, [])
    elif arch == "vgg19":
        model = vggf("E", in_channels, classes, True, precision, 0, 0, [])
    elif arch == "resnet18":
        model = resnetf("resnet18", classes, precision, 0, 0, [])
    elif arch == "resnet34":
        model = resnetf("resnet34", classes, precision, 0, 0, [])
    elif arch == "resnet50":
        model = resnetf("resnet50", classes, precision, 0, 0, [])
    elif arch == "resnet101":
        model = resnetf("resnet101", classes, precision, 0, 0, [])
    else:
        model = lenetf(in_channels, classes, precision, 0, 0, [])

    # print(model)
    checkpoint_epoch = -1

    if retrain:
        if not os.path.exists(checkpoint_path):
            for x in range(cfg.epochs, -1, -1):
                if os.path.exists(model_path_from_base(checkpoint_path, x)):
                    checkpoint_path = model_path_from_base(checkpoint_path, x)
                    break

        if not os.path.exists(checkpoint_path):
            print("Checkpoint path not exists")
            return model, checkpoint_epoch

        # print("Restoring model from checkpoint", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        # print("restored checkpoint at epoch - ", checkpoint["epoch"])
        # print("Training loss =", checkpoint["loss"])
        # print("Training accuracy =", checkpoint["accuracy"])
        checkpoint_epoch = checkpoint["epoch"]

    return model, checkpoint_epoch


def init_models_faulty(
    arch,
    in_channels,
    precision,
    retrain,
    checkpoint_path,
    faulty_layers,
    bit_error_rate,
    position,
    seed=0,
    dataset='cifar',
):

    """
    Perturbed (if needed) model loader.
    """

    if not cfg.faulty_layers or len(cfg.faulty_layers) == 0:
        return init_models(
            arch, in_channels, precision, retrain, checkpoint_path
        )
    else:
        """Perturbed models, where the weights are injected with bit
        errors at the rate of ber at the specified positions"""
        
        classes = 10
        if dataset == 'cifar10':
            classes = 10
        elif dataset == 'cifar100':
            classes = 100
        elif dataset == 'tinyimagenet':
            classes = 200
        elif dataset == 'gtsrb':
            classes = 43
        else:
            classes = 10

        if arch == "vgg11":
            model = vggf(
                "A",
                in_channels,
                classes,
                True,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif arch == "vgg16":
            model = vggf(
                "D",
                in_channels,
                classes,
                True,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif arch == "vgg19":
            model = vggf(
                "E",
                in_channels,
                classes,
                True,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )    
        elif arch == "resnet18":
            model = resnetf(
                "resnet18",
                classes,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif arch == "resnet34":
            model = resnetf(
                "resnet34",
                classes,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif arch == "resnet50":
            model = resnetf(
                "resnet50",
                classes,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif arch == "resnet101":
            model = resnetf(
                "resnet101",
                classes,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        else:
            model = lenetf(
                in_channels,
                classes,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )

    # print(model)
    checkpoint_epoch = -1

    if retrain:
        if not os.path.exists(checkpoint_path):
            for x in range(cfg.epochs, -1, -1):
                if os.path.exists(model_path_from_base(checkpoint_path, x)):
                    checkpoint_path = model_path_from_base(checkpoint_path, x)
                    break

        if not os.path.exists(checkpoint_path):
            print("Checkpoint path not exists")
            return model, checkpoint_epoch

        # print("Restoring model from checkpoint", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        # print("restored checkpoint at epoch - ", checkpoint["epoch"])
        # print("Training loss =", checkpoint["loss"])
        # print("Training accuracy =", checkpoint["accuracy"])
        checkpoint_epoch = checkpoint["epoch"]

    return model, checkpoint_epoch


def init_models_pairs(
    arch,
    in_channels,
    precision,
    retrain,
    checkpoint_path,
    faulty_layers,
    bit_error_rate,
    position,
    seed=0,
    dataset='cifar'
):

    """Load the default model as well as the corresponding perturbed model"""

    model, checkpoint_epoch = init_models(
        arch, in_channels, precision, retrain, checkpoint_path, dataset=dataset
    )
    model_p, checkpoint_epoch_p = init_models_faulty(
        arch,
        in_channels,
        precision,
        retrain,
        checkpoint_path,
        faulty_layers,
        bit_error_rate,
        position,
        seed=seed,
        dataset=dataset,
    )

    return model, checkpoint_epoch, model_p, checkpoint_epoch_p


def default_base_model_path(data_dir, arch, dataset, precision, fl, ber, pos):
    extra = [arch, dataset, "p", str(precision), "model"]
    if len(fl) != 0:
        arch = arch + "f"
        extra[0] = arch
        extra.append("fl")
        extra.append("-".join(fl))
        extra.append("ber")
        extra.append("%03.3f" % ber)
        extra.append("pos")
        extra.append(str(pos))
    return os.path.join(data_dir, arch, dataset, "_".join(extra))


def default_model_path(
    data_dir, arch, dataset, precision, fl, ber, pos, epoch
):
    extra = [arch, dataset, "p", str(precision), "model"]
    if len(fl) != 0:
        arch = arch + "f"
        extra[0] = arch
        extra.append("fl")
        extra.append("-".join(fl))
        extra.append("ber")
        extra.append("%03.3f" % ber)
        extra.append("pos")
        extra.append(str(pos))
    extra.append(str(epoch))
    return os.path.join(data_dir, arch, dataset, "_".join(extra) + ".pth")


def model_path_from_base(basename, epoch):
    return basename + "_" + str(epoch) + ".pth"
