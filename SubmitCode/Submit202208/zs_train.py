import os
import sys

import torch
import torch.optim as optim 
from torch import nn

from config import cfg
from models import default_model_path, init_models_faulty, init_models

__all__ = ["training"]

debug = False
torch.manual_seed(0)

def training(trainloader, arch, dataset, in_channels, precision, retrain, checkpoint_path, force, device, fl, ber, pos,):

    """
        Apply quantization aware training.
        :param trainloader: The loader of training data.
        :param arch: A string. The architecture of the model would be used.
        :param dataset: A string. The name of the training data.
        :param in_channels: An int. The input channels of the training data.
        :param precision: An int. The number of bits would be used to quantize
                                  the model.
        :param retrain: A boolean. Start from checkpoint.
        :param checkpoint_path: A string. The path that stores the models.
        :param force: Overwrite checkpoint.
        :param device: A string. Specify using GPU or CPU.
    """

    model, checkpoint_epoch = init_models(arch, 3, precision, retrain, checkpoint_path, dataset) # Quantization Aware Training without using bit error!

    print("Training with Learning rate %.4f" % (cfg.learning_rate))

    opt = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)

    model = model.to(device)
    from torchsummary import summary
    if dataset == 'tinyimagenet':
        summary(model, (3, 64, 64))
    else:
        summary(model, (3, 32, 32))

    torch.backends.cudnn.benchmark = True

    for x in range(checkpoint_epoch + 1, cfg.epochs):

        print("Epoch: %03d" % x)

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

            model.train()
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

            # Compute gradient of perturbed weights with perturbed loss
            loss.backward()

            # restore model weights with unquantized value
            # This step is not important because list_init_params == model.parameters()
            # Therefore, apply gradients on model.parameters() directly is OK.
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

        accuracy = running_correct.double() / (len(trainloader.dataset))
        print(
            "For epoch: {}, loss: {:.6f}, accuracy: {:.5f}".format(
                x, running_loss / len(trainloader.dataset), accuracy
            )
        )
        if (x+1)%10 == 0:

            model_path = default_model_path(
                cfg.model_dir, arch, dataset, precision, fl, ber, pos, x+1
            )

            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))

            if os.path.exists(model_path) and not force:
                print("Checkpoint already present ('%s')" % model_path)
                sys.exit(1)

            torch.save(
                {
                    "epoch": x,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": running_loss / batch_id,
                    "accuracy": accuracy,
                },
                model_path,
            )
