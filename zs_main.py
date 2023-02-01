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

# Single model
import zs_test as test
import zs_train as train
# import zs_train_sparse as sptrain

import zs_train_input_transform_sparse as sparse

import zs_train_input_transform_single as transform_single
import zs_train_input_transform_single_gen as transform_single_gen
import zs_train_input_transform_single_attention as transform_single_attention
import zs_train_sam as train_sam

import zs_train_input_transform_activation as transform_activation

# EOPM-Based
import zs_train_input_transform_eopm as transform_eopm
import zs_train_input_transform_mlp_eopm as transform_mlp_eopm
import zs_train_input_transform_eopm_gen as transform_eopm_gen

# Adversarial-Based
import zs_train_input_transform_adversarial as transform_adversarial
import zs_train_input_transform_adversarial_gen as transform_adversarial_gen
import zs_train_input_transform_adversarial_gen_bit as transform_adversarial_gen_bit
import zs_train_input_transform_mlp_adversarial as transform_mlp_adversarial
import zs_train_input_transform_adversarial_w as transform_adversarial_w
import zs_train_input_transform_eval as transform_eval
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
        choices=["resnet18", "resnet34", "resnet50", "resnet101",
                 "vgg11", "vgg16", "vgg19", "mobilenetv2", "lenet"],
        default="resnet18",
    )
    parser.add_argument(
        "mode",
        help="Specify operation to perform",
        default="eval",
        choices=["train", "eval", "transform_single", "transform_eval", "transform_activation",
                "transform_single_attention",
                "transform_single_gen","transform_eopm_gen",
                 "transform_eopm", "transform_mlp_eopm", 
                 "transform_adversarial", "transform_mlp_adversarial", "transform_adversarial_w", 
                 "transform_adversarial_gen", "transform_adversarial_gen_bit", "sparse", "train_sam"
                ],
    )
    parser.add_argument(
        "dataset",
        help="Specify dataset",
        choices=["cifar10", "cifar100", "mnist", "fashion", "tinyimagenet", "gtsrb"],
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
        "-LR",
        "--learning_rate",
        type=float,
        help="Learning rate for training input transformation of training clean model.",
        default=1e-2,
    )
    group.add_argument(
        "-LM",
        "--lambdaVal",
        type=float,
        help="Lambda value between two loss function",
        default=1,
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
    group.add_argument(
        "-N",
        "--N_perturbed_model",
        type=int,
        help="How many perturbed model will be used for training.",
        default=100,
    )
    group.add_argument(
        "-G",
        "--Generator",
        type=str,
        help="Which generator to be used.",
        default='large',
    )
    group.add_argument(
        "-PGD",
        "--pgd_step",
        type=int,
        help="How many pgd steps for training.",
        default=2,
    )

    args = parser.parse_args()
    #cfg.epochs = args.epochs
    #cfg.learning_rate = args.learning_rate
    #cfg.batch_size = args.batch_size
    #cfg.test_batch_size = args.test_batch_size
    #cfg.lb = args.lambdaVal
    #cfg.N = args.N_perturbed_model
    #cfg.G = args.Generator
    #cfg.PGD_STEP = args.pgd_step
    #cfg.G = args.Generator


    # if args.position>args.precision-1:
    #    print('ERROR: specified bit position for error exceeds the precision')
    #    exit(0)

    print("Preparing data..", args.dataset)
    if args.dataset == "cifar10":
        dataset = "cifar10"
        in_channels = 3
        transform_train = transforms.Compose(
            [
                # transforms.RandomAffine(degrees = 0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                #transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                #transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root=cfg.data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
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
    elif args.dataset == "cifar100":
        dataset = "cifar100"
        in_channels = 3
        transform_train = transforms.Compose(
            [
                # transforms.RandomAffine(degrees = 0, translate=(0.35, 0.35), scale=(0.65, 1.35)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.CIFAR100(
            root=cfg.data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.CIFAR100(
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
    elif args.dataset == "gtsrb":
        dataset = "gtsrb"
        in_channels = 3
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomAffine(degrees = 0, translate=(0.35, 0.35), scale=(0.65, 1.35)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.GTSRB(
            root=cfg.data_dir,
            split="train",
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.GTSRB(
            root=cfg.data_dir,
            split='test',
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )
    elif args.dataset == "tinyimagenet":
        dataset = "tinyimagenet"
        in_channels = 3
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(cfg.data_dir + '/tiny-imagenet-200', 'train'), 
            transform=transform_train)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(cfg.data_dir + '/tiny-imagenet-200', 'val'), 
            transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )

    elif args.dataset == "mnist":
        dataset = "mnist"
        in_channels = 1
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
        in_channels = 1
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

    if args.checkpoint is None and args.mode != "transform":
        args.checkpoint = default_base_model_path(
            cfg.data_dir,
            args.arch,
            dataset,
            cfg.precision,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.checkpoint is None and args.mode == "transform":
        args.checkpoint = []
        args.checkpoint.append(
            default_base_model_path(
                cfg.data_dir,
                args.arch,
                dataset,
                cfg.precision,
                [],
                args.bit_error_rate,
                args.position,
            )
        )
        args.checkpoint.append(
            default_base_model_path(
                cfg.data_dir,
                args.arch,
                dataset,
                cfg.precision,
                cfg.faulty_layers,
                args.bit_error_rate,
                args.position,
            )
        )

    if args.mode == "train":
        print("training args", args)
        train.training(
            trainloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.retrain,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_single":
        print("input_transform_train_single", args)
        cfg.save_dir = 'single_p/'
        cfg.save_dir_curve = 'single_curve/'
        transform_single.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_single_gen":
        print("input_transform_train_single_gen", args)
        cfg.save_dir = 'single_p_gen/'
        cfg.save_dir_curve = 'single_curve_gen/'
        transform_single_gen.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_single_attention":
        print("input_transform_train_single_attention", args)
        cfg.save_dir = 'single_p_attention/'
        cfg.save_dir_curve = 'single_curve_attention/'
        transform_single_attention.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_activation":
        print("input_transform_train_activation", args)
        cfg.save_dir = 'activation_p/'
        cfg.save_dir_curve = 'activation_curve/'
        transform_activation.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_eopm":
        print("input_transform_train_eopm", args)
        cfg.save_dir = 'eopm_p/'
        cfg.save_dir_curve = 'eopm_curve/'
        transform_eopm.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_eopm_gen":
        print("input_transform_train_eopm_gen", args)
        cfg.save_dir = 'eopm_p_gen/'
        cfg.save_dir_curve = 'eopm_curve_gen/'
        transform_eopm_gen.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_mlp_eopm":
        print("input_transform_train_mlp_eopm", args)
        cfg.save_dir = 'mlp_eopm_p_w/'
        cfg.save_dir_curve = 'mlp_eopm_curve_w/'
        transform_mlp_eopm.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_adversarial":
        print("input_transform_train_adversarial", args)
        cfg.save_dir = 'adversarial_p/'
        cfg.save_dir_curve = 'adversarial_curve/'
        transform_adversarial.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_mlp_adversarial":
        print("input_transform_train_mlp_adversarial", args)
        cfg.save_dir = 'mlp_adversarial_p/'
        cfg.save_dir_curve = 'mlp_adversarial_curve/'
        transform_mlp_adversarial.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_adversarial_gen":
        print("input_transform_train_adversarial_gen", args)
        cfg.save_dir = 'adversarial_gen/'
        cfg.save_dir_curve = 'adversarial_gen/'
        transform_adversarial_gen.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_adversarial_gen_bit":
        print("input_transform_train_adversarial_gen_bit", args)
        cfg.save_dir = 'adversarial_gen_bit/'
        cfg.save_dir_curve = 'adversarial_gen_bit/'
        transform_adversarial_gen_bit.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_adversarial_w":
        print("input_transform_train_adversarial_w", args)
        cfg.save_dir = 'adversarial_p_w/'
        cfg.save_dir_curve = 'adversarial_curve_w/'
        transform_adversarial_w.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_eval":
        print("input_transform_train_eval", args)
        transform_eval.transform_eval(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "sparse":
        print("input_transform_train_sparse", args)
        sparse.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "eval":
        print("test model", args)
        test.inference(
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "train_sam":
        print("train model with SAM", args)
        train_sam.training(
            trainloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.retrain,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
