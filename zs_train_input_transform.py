from config import cfg

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os
import argparse
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torchvision.models as models
import time

import sys
sys.path.append('./models')
from models import lenetf
from models import vggf
from models import resnetf

sys.path.append('./faultmodels')
from faultmodels import randomfault
import pdb
torch.manual_seed(0)


class Program(nn.Module):
  '''
      Apply reprogramming.
  '''

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
  labels = labels.view(labels.size(0))  ## changing the size from (batch_size,1) to batch_size. 
  loss = nn.CrossEntropyLoss()(model_outputs, labels)
  return loss, preds


def init_models(arch, in_channels, bit_error_rate, precision, position, faulty_layers, checkpoint_path, seed=0):
  """ unperturbed model 
  """
  if arch == 'vgg11':
    model  = vggf('A',in_channels, 10, True, precision, 0,0, 0,0,[])
  elif arch == 'vgg16':
    model  = vggf('D',in_channels, 10, True, precision, 0, 0 ,0,0,[])
  elif arch == 'resnet18':
    model = resnetf('resnet18', 10, precision, 0,0,0,0,[])
  elif arch == 'resnet34':
    model = resnetf('resnet34', 10, precision, 0, 0,0,0,[])
  else:
    model = lenetf(in_channels,10,precision, 0,0,0,0,[])
  
  """ Perturbed model, where the weights are injected with bit errors at the rate of ber
  """
  rf = randomfault.RandomFaultModel(bit_error_rate, precision, position, seed)
  BitErrorMap0 = torch.tensor(rf.BitErrorMap_flip0).to(torch.int32).to(device)
  BitErrorMap1 = torch.tensor(rf.BitErrorMap_flip1).to(torch.int32).to(device)
  if arch == 'vgg11':
    model_p  = vggf('A',in_channels, 10, True, precision, bit_error_rate, position, BitErrorMap0, BitErrorMap1, faulty_layers)
  elif arch == 'vgg16':
    model_p  = vggf('D',in_channels, 10, True, precision, bit_error_rate, position, BitErrorMap0, BitErrorMap1, faulty_layers)
  elif arch == 'resnet18':
    model_p = resnetf('resnet18', 10, precision, bit_error_rate, position, BitErrorMap0, BitErrorMap1, faulty_layers)
  elif arch == 'resnet34':
    model_p = resnetf('resnet34', 10, precision, bit_error_rate, position, BitErrorMap0, BitErrorMap1, faulty_layers)
  else:
    model_p = lenetf(in_channels,10,precision, bit_error_rate, position, BitErrorMap0, BitErrorMap1, faulty_layers)

  model = model.to(device)
  model_p = model_p.to(device)


  print('Restoring model from checkpoint', checkpoint_path)
  checkpoint = torch.load(checkpoint_path)

  model.load_state_dict(checkpoint['model_state_dict'])
  print('restored checkpoint at epoch - ',checkpoint['epoch'])
  print('Training loss =', checkpoint['loss'])
  print('Training accuracy =', checkpoint['accuracy'])
  checkpoint_epoch=checkpoint['epoch']

  model_p.load_state_dict(checkpoint['model_state_dict'])

  return model, model_p

def accuracy_checking(model_orig, model_p, trainloader, testloader, pg, device):
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
    x_adv = pg(x)
    out_orig = model_orig(x_adv)
    out_p = model_p(x_adv)
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
    x_adv = pg(x)
    out_orig = model_orig(x_adv)
    out_p = model_p(x_adv)
    _, pred_orig = out_orig.max(1)
    _, pred_p = out_p.max(1)
    y = y.view(y.size(0))
    correct_orig_test += torch.sum(pred_orig == y.data).item()
    correct_p_test += torch.sum(pred_p == y.data).item()
  accuracy_orig_test = correct_orig_test / (len(testloader.dataset))
  accuracy_p_test = correct_p_test / (len(testloader.dataset))

  print('Accuracy of training data: clean model: {:5f}, perturbed model: {:5f}'.format(accuracy_orig_train,
                                                                                       accuracy_p_train))
  print('Accuracy of testing data: clean model: {:5f}, perturbed model: {:5f}'.format(accuracy_orig_test,
                                                                                      accuracy_p_test))


def transform_train(trainloader, testloader, in_channels, arch, dataset, ber, precision, position, checkpoint_path, device):
  '''
      Apply quantization aware training.
      :param trainloader: The loader of training data.
      :param in_channels: An int. The input channels of the training data.
      :param arch: A string. The architecture of the model would be used.
      :param dataset: A string. The name of the training data.
      :param ber: A float. How many rate of bits would be attacked.
      :param precision: An int. The number of bits would be used to quantize the model.
      :param position:
      :param checkpoint_path: A string. The path that stores the models.
      :param device: Specify GPU usage.
  '''
  #model = torch.nn.DataParallel(model)
  torch.backends.cudnn.benchmark = True

  model, model_perturbed = init_models(arch, in_channels, ber, precision, position, cfg.faulty_layers, checkpoint_path,
                                       cfg.seed)
  Pg = Program(cfg)
  model, model_perturbed, Pg = model.to(device), model_perturbed.to(device), Pg.to(device)

  model.eval()
  model_perturbed.eval()
  Pg.train()

  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Pg.parameters()),lr=cfg.learning_rate, betas=(0.5, 0.999), weight_decay=5e-4)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=cfg.decay)
  lb = cfg.lb  # Lambda

  print('========== Start checking the accuracy before applying input transform ==========')
  accuracy_checking(model, model_perturbed, trainloader, testloader, Pg, device)

  for name, param in Pg.named_parameters():
    print('Param name: {}, grads is: {}'.format(name, param.requires_grad))

  print('========== Start training the parameter of the input transform ==========')
  for epoch in range(cfg.epochs):
    running_loss = 0
    running_correct_orig = 0
    running_correct_p = 0
    total = 0
    for batch_id, (image, label) in enumerate(trainloader):
      total += 1
      image, label = image.to(device), label.to(device)
      image_adv = Pg(image)
      out, out_biterror = model(image_adv), model_perturbed(image_adv)
      loss_orig, pred_orig = compute_loss(out, label)
      loss_p, pred_p = compute_loss(out_biterror, label)
      loss = loss_orig + lb * loss_p

      optimizer.zero_grad()
      # Pg.zero_grad()
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
      running_loss += loss.item()
      running_correct_orig += torch.sum(pred_orig == label.data).item()
      running_correct_p += torch.sum(pred_p == label.data).item()

    accuracy_orig = running_correct_orig / (len(trainloader.dataset))
    accuracy_p = running_correct_p / (len(trainloader.dataset))
    print(
      'For epoch: {}, loss: {:.6f}, accuracy for clean model: {:.5f}, accuracy for perturbed model: {:.5f}'.format(
        epoch + 1, running_loss / len(trainloader.dataset), accuracy_orig, accuracy_p))

    # if (epoch + 1) % 20 == 0 or (epoch + 1) == cfg.epochs:
    #   torch.save({'Reprogrammed Perturbation': Pg.P}, '{}arch_{}_W_{}.pt'.format(cfg.save_dir, arch, epoch + 1))

    #self.validate()
    # if x%20==19:
    #   lb *= 0.5
    #   print("Lambda value: ", lb)

  print('========== Start checking the accuracy after applying input transform ==========')
  accuracy_checking(model, model_perturbed, trainloader, testloader, Pg, device)


