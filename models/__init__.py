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

from .lenet import lenet  # noqa: F401
from .lenetf import lenetf  # noqa: F401
from .resnet import resnet  # noqa: F401
from .resnetf import resnetf  # noqa: F401
from .simplenet import simplenet  # noqa: F401
from .vgg import vgg  # noqa: F401
from .vggf import vggf  # noqa: F401


def init_models(arch, precision, retrain, checkpoint_path):

    """
    Default model loader
    """

    in_channels = 3

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

        print("Restoring model from checkpoint", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        print("restored checkpoint at epoch - ", checkpoint["epoch"])
        print("Training loss =", checkpoint["loss"])
        print("Training accuracy =", checkpoint["accuracy"])
        checkpoint_epoch = checkpoint["epoch"]

    return model, checkpoint_epoch


def init_models_faulty(
    arch,
    precision,
    retrain,
    checkpoint_path,
    faulty_layers,
    bit_error_rate,
    position,
):

    """
    Perturbed (if needed) model loader.
    """

    in_channels = 3

    if not cfg.faulty_layers or len(cfg.faulty_layers) == 0:
        return init_models(arch, precision, retrain, checkpoint_path)
    else:
        """Perturbed models, where the weights are injected with bit
        errors at the rate of ber at the specified positions"""
        rf = randomfault.RandomFaultModel(
            bit_error_rate, precision, position, 0
        )
        BitErrorMap0 = (
            torch.tensor(rf.BitErrorMap_flip0).to(torch.int32).to(cfg.device)
        )
        BitErrorMap1 = (
            torch.tensor(rf.BitErrorMap_flip1).to(torch.int32).to(cfg.device)
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

        print("Restoring model from checkpoint", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        print("restored checkpoint at epoch - ", checkpoint["epoch"])
        print("Training loss =", checkpoint["loss"])
        print("Training accuracy =", checkpoint["accuracy"])
        checkpoint_epoch = checkpoint["epoch"]

    return model, checkpoint_epoch


def init_models_pairs(
    arch,
    precision,
    retrain,
    checkpoint_path,
    faulty_layers,
    bit_error_rate,
    position,
):

    """Load the default model as well as the corresponding perturbed model"""

    model, checkpoint_epoch = init_models(
        arch, precision, retrain, checkpoint_path[0]
    )
    model_p, checkpoint_epoch_p = init_models_faulty(
        arch,
        precision,
        retrain,
        checkpoint_path[1],
        faulty_layers,
        bit_error_rate,
        position,
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
