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
from models import init_models_faulty, init_models

# import loss_landscapes
# import loss_landscapes.metrics

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

debug = False
visualize = False


def inference(
    testloader,
    arch,
    dataset,
    in_channels,
    precision,
    checkpoint_path,
    device,
    faulty_layers,
    ber,
    position,
):
    # print(torch.version.cuda)
    # print(torch.__version__)
    
    model, checkpoint_epoch = init_models_faulty(
        arch,
        in_channels,
        precision,
        True,
        checkpoint_path,
        faulty_layers,
        ber,
        position,
    )

    if arch == "resnet18" or arch == "resnet34":
        stats.resnet_register_hooks(model, arch)
    elif arch == "vgg16" or arch == "vgg11":
        stats.vgg_register_hooks(model, arch)
    else:
        print("Inspection/Results hooks not implemented for: %s" % arch)

    logger = stats.DataLogger(
        int(len(testloader.dataset) / testloader.batch_size),
        testloader.batch_size,
    )

    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

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
    
    # STEPS = 100
    # x, y = iter(testloader).__next__()
    # criterion = torch.nn.CrossEntropyLoss()
    # metric = loss_landscapes.metrics.Loss(criterion, x, y)
    # loss_data = loss_landscapes.random_plane(model, metric, 10, STEPS, normalization='filter')
    
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    # Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    # ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.set_title('Surface Plot of Loss Landscape')
    # plt.savefig('loss_landscape.png')


