import torchvision.transforms as transforms
import torchvision
import zs_train_input_transform as transform
import zs_test as test
import zs_train as train
from config import cfg
import torch

import sys
import argparse
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


sys.path.append("./faultmodels")

sys.path.append("./models")

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(argv):

    print("Running command:", str(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arch",
        help="<resnet18> <resnet34> <vgg11> <vgg16> <lenet> "
        "specify network architecture",
        default="resnet18",
    )
    parser.add_argument(
        "mode", help="<train> <prune> <eval> specify operation", default="eval"
    )
    parser.add_argument(
        "dataset",
        help="<fashion> <cifar10> <mnist> specify dataset",
        default="fashion",
    )
    parser.add_argument(
        "-ber",
        "--bit_error_rate",
        type=float,
        help="bit error rate for training corresponding to known voltage",
        default=0.01,
    )
    parser.add_argument(
        "-pos",
        "--position",
        type=int,
        help="specify position of bit errors",
        default=-1,
    )
    parser.add_argument(
        "-rt",
        "--retrain",
        action="store_true",
        help="retrain on top of already trained model",
        default=False,
    )
    parser.add_argument(
        "-cp",
        "--checkpoint",
        help="Name of the stored model that needs to be "
        "retrained or used for test",
        default="./resnet18_checkpoints/resnet_resnet_model_119.pth",
    )
    args = parser.parse_args()

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
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
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

    if args.mode == "train":
        print("training args", args)
        train.training(
            trainloader,
            args.arch,
            dataset,
            cfg.precision,
            args.retrain,
            args.checkpoint,
            device,
        )
    elif args.mode == "transform":

        transform.transform_train(
            trainloader,
            args.arch,
            dataset,
            args.bit_error_rate,
            cfg.precision,
            args.position,
            args.checkpoint,
            mean,
            std,
            device,
        )
    else:
        test.inference(
            testloader,
            args.arch,
            dataset,
            args.bit_error_rate,
            cfg.precision,
            args.position,
            args.checkpoint,
            cfg.faulty_layers,
            device,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
