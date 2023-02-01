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

from config import cfg
from models import init_models_pairs
import numpy as np

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


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


def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(
        labels.size(0)
    )  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds


def accuracy_checking(
    model_orig, model_p, trainloader, testloader, pg, device, use_transform=False
):
    # For training data first:
    total_train = 0
    total_test = 0
    correct_orig_train = 0
    correct_p_train = 0
    correct_orig_test = 0
    correct_p_test = 0
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

    return accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test


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
    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    (
        model,
        checkpoint_epoch,
        model_perturbed,
        checkpoint_epoch_perturbed,
    ) = init_models_pairs(
        arch,
        in_channels,
        precision,
        True,
        checkpoint_path,
        fl,
        ber,
        pos,
        seed=seed,
    )

    assert checkpoint_epoch == checkpoint_epoch_perturbed

    Pg = Program(cfg)
    model, model_perturbed, Pg = (
        model.to(device),
        model_perturbed.to(device),
        Pg.to(device),
    )

    model.eval()
    model_perturbed.eval()
    Pg.train()

    #optimizer = torch.optim.Adam(
    #    filter(lambda p: p.requires_grad, Pg.parameters()),
    #    lr=cfg.learning_rate,
    #    betas=(0.5, 0.999),
    #    weight_decay=5e-4,
    #)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, Pg.parameters()), lr=cfg.learning_rate, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=4, gamma=cfg.decay
    )
    lb = cfg.lb  # Lambda

    #print(
    #    "========== Start checking the accuracy "
    #    "before applying input transform =========="
    #)
    #accuracy_checking(
    #    model, model_perturbed, trainloader, testloader, Pg, device
    #)

    for name, param in Pg.named_parameters():
        print("Param name: {}, grads is: {}".format(name, param.requires_grad))

    print(
        "========== Start training the parameter"
        " of the input transform by using EOT attack =========="
    )
    for epoch in range(cfg.epochs):
        running_loss = 0
        running_loss_forkeepclean = 0
        running_correct_orig = 0
        running_correct_p = 0
              
        # For each epoch, we will use N perturbed model for training.
        for batch_id, (image, label) in enumerate(trainloader):
            total_loss = 0  
            image, label = image.to(device), label.to(device)
            image_adv = Pg(image)  # pylint: disable=E1102
            for j in range(cfg.N):
                # print("*** For using the {}-th perturbed model ***".format(j))
                (model, checkpoint_epoch, model_perturbed, checkpoint_epoch_perturbed) = init_models_pairs( 
                    arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=j)
                #model = nn.DataParallel(model)
                #model_perturbed = nn.DataParallel(model_perturbed)
                model, model_perturbed = model.to(device), model_perturbed.to(device)
                # We only need to inference clean model for one time!
                if j == 0: 
                    model.eval()
                    out = model(image_adv)  # pylint: disable=E1102
                    loss_orig, pred_orig = compute_loss(out, label)
                    running_correct_orig += torch.sum(pred_orig == label.data).item()
                    running_loss_forkeepclean += loss_orig.item()

                model_perturbed.eval()
                out_biterror = model_perturbed(image_adv)  # pylint: disable=E1102
                loss_p, pred_p = compute_loss(out_biterror, label)
                total_loss += loss_p
                # running_loss += loss_p.item()
                running_correct_p += torch.sum(pred_p == label.data).item()
            
            total_loss /= cfg.N # Average the loss. Only perturbed loss is in this total_loss now!
            total_loss = loss_orig + lb * total_loss
            running_loss += total_loss.item()
            optimizer.zero_grad()
            # Pg.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            #print(total_loss.item())
            #print(Pg.P.grad)
            
            

        accuracy_orig = running_correct_orig / len(trainloader.dataset)
        accuracy_p = running_correct_p / (len(trainloader.dataset) * cfg.N)
        print(
            "For epoch: {}, loss: {:.6f}, accuracy for {} clean model:"
            "{:.5f}, accuracy for {} perturbed model: {:.5f}".format(
                epoch + 1,
                running_loss,
                cfg.N,
                accuracy_orig,
                cfg.N,
                accuracy_p,
            )
        )

        # if (epoch + 1) % 20 == 0 or (epoch + 1) == cfg.epochs:
        #   torch.save({'Reprogrammed Perturbation': Pg.P},
        #   '{}arch_{}_W_{}.pt'.format(cfg.save_dir, arch, epoch + 1))

        # self.validate()
        # if x%20==19:
        #   lb *= 0.5
        #   print("Lambda value: ", lb)

    print('========== Start checking the accuracy with different perturbed model ==========')
    # Setting without input transformation
    accuracy_orig_train_list = []
    accuracy_p_train_list = []
    accuracy_orig_test_list = []
    accuracy_p_test_list = []

    # Setting with input transformation
    accuracy_orig_train_list_with_transformation = []
    accuracy_p_train_list_with_transformation = []
    accuracy_orig_test_list_with_transformation = []
    accuracy_p_test_list_with_transformation = []

    for i in range(1000, 1010):
        print(' ********** For seed: {} ********** '.format(i))
        (model, checkpoint_epoch, model_perturbed, checkpoint_epoch_perturbed) = init_models_pairs( 
                    arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=i)
        model, model_perturbed = model.to(device), model_perturbed.to(device),
        model.eval()
        model_perturbed.eval()

        # Without using transform
        accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Pg, device, use_transform=False)
        accuracy_orig_train_list.append(accuracy_orig_train)
        accuracy_p_train_list.append(accuracy_p_train)
        accuracy_orig_test_list.append(accuracy_orig_test)
        accuracy_p_test_list.append(accuracy_p_test)

        # With input transform
        accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Pg, device, use_transform=True)
        accuracy_orig_train_list_with_transformation.append(accuracy_orig_train)
        accuracy_p_train_list_with_transformation.append(accuracy_p_train)
        accuracy_orig_test_list_with_transformation.append(accuracy_orig_test)
        accuracy_p_test_list_with_transformation.append(accuracy_p_test)
  
    # Without using transform
    print('The average results without input transformation -> accuracy_orig_train: {:5f}, accuracy_p_train: {:5f}, accuracy_orig_test: {:5f}, accuracy_p_test: {:5f}'.format(
      np.mean(accuracy_orig_train_list), np.mean(accuracy_p_train_list), np.mean(accuracy_orig_test_list), np.mean(accuracy_p_test_list)
    ))
    print('The average results without input transformation -> std_accuracy_orig_train: {:5f}, std_accuracy_p_train: {:5f}, std_accuracy_orig_test: {:5f}, std_accuracy_p_test: {:5f}'.format(
      np.std(accuracy_orig_train_list), np.std(accuracy_p_train_list), np.std(accuracy_orig_test_list), np.std(accuracy_p_test_list)
    ))

    print()

    # With input transform
    print('The average results with input transformation -> accuracy_orig_train: {:5f}, accuracy_p_train: {:5f}, accuracy_orig_test: {:5f}, accuracy_p_test: {:5f}'.format(
      np.mean(accuracy_orig_train_list_with_transformation), np.mean(accuracy_p_train_list_with_transformation), np.mean(accuracy_orig_test_list_with_transformation), np.mean(accuracy_p_test_list_with_transformation)
    ))
    print('The average results with input transformation -> std_accuracy_orig_train: {:5f}, std_accuracy_p_train: {:5f}, std_accuracy_orig_test: {:5f}, std_accuracy_p_test: {:5f}'.format(
      np.std(accuracy_orig_train_list_with_transformation), np.std(accuracy_p_train_list_with_transformation), np.std(accuracy_orig_test_list_with_transformation), np.std(accuracy_p_test_list_with_transformation)
    ))

    '''
    print(
        "========== Start checking the accuracy "
        "after applying input transform =========="
    )
    accuracy_checking(
        model, model_perturbed, trainloader, testloader, Pg, device
    )
    '''
