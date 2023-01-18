import os
import sys

import torch
import torch.optim as optim 
from torch import nn

from config import cfg
from models import default_model_path, init_models_faulty, init_models

import torch.nn.utils.prune as prune
from quantized_ops import zs_quantized_ops

__all__ = ["training"]

debug = False
torch.manual_seed(0)


def training(
    trainloader, arch, dataset, in_channels, precision, retrain, checkpoint_path, force, device, fl, ber, pos,
):

    model, checkpoint_epoch = init_models(arch, 3, precision, retrain, checkpoint_path, dataset) # Quantization Aware Training without using bit error!

    print("Training with Learning rate %.4f" % (cfg.learning_rate))

    model = model.to(device)
    torch.backends.cudnn.benchmark = True

    for x in range(checkpoint_epoch + 1, cfg.epochs):

        print("Epoch: %03d" % x)

        running_loss = 0.0
        running_correct = 0
        for batch_id, (inputs, outputs) in enumerate(trainloader):
            
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            outputs = outputs.view(outputs.size(0)) # changing the size from (batch_size,1) to batch_size.

            model.zero_grad()

            model.train()
            
            model_outputs = model(inputs)  # pylint: disable=E1102

            loss = nn.CrossEntropyLoss()(model_outputs, outputs)

            # compute gradient of perturbed weights with perturbed loss
            loss.backward()
            
            # record first backprop stats
            _, preds = torch.max(model_outputs, 1)
            running_loss += loss.item()
            running_correct += torch.sum(preds == outputs.data)

            # calculate e_w and store in weights
            e_w_clone = []
            with torch.no_grad():
                for param in model.parameters():
                    numer = 0.05 * param.grad
                    denom = torch.linalg.norm(param.grad, dim = 0, ord = 2)
                    denom = torch.linalg.norm(denom, dim = 0, ord = 2)
                    denom = torch.linalg.norm(denom, dim = 0, ord = 2)
                    e_w = numer / denom
                    e_w = e_w.cuda()
                    e_w_clone.append(e_w)
                    param += e_w
                
            model.zero_grad()
            
            # forward pass weights + e_w
            model_outputs = model(inputs)  # pylint: disable=E1102
            
            # backprop again
            loss = nn.CrossEntropyLoss()(model_outputs, outputs)
            
            # compute gradient of weights + e(w)
            loss.backward()
            
            # stochastic gradient descent
            with torch.no_grad():
                i = 0
                for param in model.parameters():
                    param -= cfg.learning_rate * param.grad
                    param -= e_w_clone[i]
                    i += 1
                
            
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
                    # "optimizer_state_dict": opt.state_dict(),
                    "loss": running_loss / batch_id,
                    "accuracy": accuracy,
                },
                model_path,
            )
