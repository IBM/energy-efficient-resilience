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

import argparse
import os
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import zs_test as test
import zs_train as train
import zs_train_input_transform as transform
from config import cfg
from models import default_base_model_path

np.set_printoptions(threshold=sys.maxsize)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():

    print("Running command:", str(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arch",
        help="Input network architecture",
        choices=["resnet18", "resnet34", "vgg11", "vgg16", "lenet"],
        default="resnet18",
    )
    parser.add_argument(
        "mode",
        help="Specify operation to perform",
        default="eval",
        choices=["train", "transform", "eval"],
    )
    parser.add_argument(
        "dataset",
        help="Specify dataset",
        choices=["cifar10", "mnist", "fashion"],
        default="fashion",
    )
    group = parser.add_argument_group(
        "Reliability/Error control Options",
        "Options to control the fault injection details.",
    )
    group.add_argument(
        "-ber",
        "--bit_error_rate",
        type=float,
        help="Bit error rate for training corresponding to known voltage.",
        default=0.01,
    )
    group.add_argument(
        "-pos",
        "--position",
        type=int,
        help="Position of bit errors.",
        default=-1,
    )
    group = parser.add_argument_group(
        "Initialization options", "Options to control the initial state."
    )
    group.add_argument(
        "-rt",
        "--retrain",
        action="store_true",
        help="Continue training on top of already trained model."
        "It will start the "
        "process from the provided checkpoint.",
        default=False,
    )
    group.add_argument(
        "-cp",
        "--checkpoint",
        help="Name of the stored checkpoint that needs to be "
        "retrained or used for test (only used if -rt flag is set).",
        default=None,
    )
    group.add_argument(
        "-F",
        "--force",
        action="store_true",
        help="Do not fail if checkpoint already exists. Overwrite it.",
        default=False,
    )
    group = parser.add_argument_group(
        "Other options", "Options to control training/validation process."
    )
    group.add_argument(
        "-E",
        "--epochs",
        type=int,
        help="Maxium number of epochs to train.",
        default=5,
    )
    group.add_argument(
        "-BS",
        "--batch-size",
        type=int,
        help="Training batch size.",
        default=128,
    )
    group.add_argument(
        "-TBS",
        "--test-batch-size",
        type=int,
        help="Test batch size.",
        default=100,
    )

    args = parser.parse_args()
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.test_batch_size = args.test_batch_size

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    # if args.position>args.precision-1:
    #    print('ERROR: specified bit position for error exceeds the precision')
    #    exit(0)

    print("Preparing data..", args.dataset)
    if args.dataset == "cifar10":
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
        dataset = "cifar"
        # in_channels = 3
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root=cfg.data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root=cfg.data_dir,
            train=False,
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )

    elif args.dataset == "mnist":
        dataset = "mnist"
        # in_channels = 1
        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        trainset = torchvision.datasets.MNIST(
            root=cfg.data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.MNIST(
            root=cfg.data_dir,
            train=False,
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )

    else:
        dataset = "fashion"
        # in_channels = 1
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.2860,), (0.3530,)
                ),  # per channel means and std devs
            ]
        )

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2868,), (0.3524,))]
        )

        trainset = torchvision.datasets.FashionMNIST(
            root=cfg.data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.FashionMNIST(
            root=cfg.data_dir,
            train=False,
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )

    print("Device", device)
    cfg.device = device

    assert isinstance(cfg.faulty_layers, list)

    if args.checkpoint is None:
        args.checkpoint = default_base_model_path(
            cfg.data_dir,
            args.arch,
            dataset,
            cfg.precision,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )

    if args.mode == "train":
        print("training args", args)
        train.training(
            trainloader,
            args.arch,
            dataset,
            cfg.precision,
            args.retrain,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform":
        print("input_transform_train", args)
        transform.transform_train(
            trainloader,
            args.arch,
            dataset,
            cfg.precision,
            args.retrain,
            args.checkpoint,
            args.force,
            device,
            args.faulty_layers,
            args.bit_error_rate,
            args.position,
            mean,
            std,
        )
    elif args.mode == "eval":
        print("test model", args)
        test.inference(
            testloader,
            args.arch,
            dataset,
            cfg.precision,
            args.checkpoint,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
