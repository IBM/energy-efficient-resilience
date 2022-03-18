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

torch.manual_seed(0)


class Program(nn.Module):
  def __init__(self, cfg, model, model_p, mean, std):
    super(Program, self).__init__()
    self.cfg = cfg
    self.num_classes = 10
    self.net = model
    self.net_biterror = model_p
    self.mean = Parameter(torch.from_numpy(mean).to(device), requires_grad=False)
    self.std = Parameter(torch.from_numpy(std).to(device), requires_grad=False)

    self.init_mask()
    self.W = Parameter((torch.randn(self.M.shape) * 2 - 1).to(device) * 0.0001, requires_grad=True)
    # self.W = Parameter(torch.zeros(self.M.shape), requires_grad=True)
    # self.W.data = torch.load('train_log_resnet18/lb_2/W_030.pt', map_location=torch.device(device))['W']
    #self.W = Parameter(torch.load('train_log_resnet18/lb_5e-1/W_030.pt', map_location=torch.device(device)), requires_grad=True)

    self.beta = 22
    self.temperature = self.cfg.temperature
    self.activation_ = torch.nn.Tanh()

  # Initialize mask to all 1's
  def init_mask(self):
    M = torch.ones(self.cfg.channels, self.cfg.h1, self.cfg.w1).to(device)
    self.M = Parameter(M, requires_grad=False)

  def imagenet_label2_mnist_label(self, imagenet_label):
    return imagenet_label[:, :self.num_classes]/self.temperature


  def forward(self, image):
    #X = image.data.new(self.cfg.batch_size, self.cfg.channels, self.cfg.h1, self.cfg.w1)
    #X[:] = 0
    #X[:, :, int((self.cfg.h1 - self.cfg.h2) // 2):int((self.cfg.h1 + self.cfg.h2) // 2),
    #    int((self.cfg.w1 - self.cfg.w2) // 2):int((self.cfg.w1 + self.cfg.w2) // 2)] = image.data.clone()
    X = image.data.clone()
    #X = Variable(X, requires_grad=True)
    P = self.W #self.dropout(self.W)
    X_adv = 2 * X - 1
    X_adv = torch.tanh(0.5 * (torch.log(1 + X_adv + 1e-15) - torch.log(1 - X_adv + 1e-15)) + P)
    X_adv = 0.5 * X_adv + 0.5
    print(X_adv.size())
    # X_adv = torch.clamp(X+P, 0.0, 1.0)
    X_adv = (X_adv - self.mean) / self.std

    Y = self.net(X_adv)
    Y_biterror = self.net_biterror(X_adv)

    return Y, Y_biterror



def compute_loss(model_outputs, labels):
  _, preds = torch.max(model_outputs, 1)
  outputs = outputs.view(outputs.size(0))  ## changing the size from (batch_size,1) to batch_size. 
  loss = nn.CrossEntropyLoss()(model_outputs, labels)
  return loss


def init_models(arch, bit_error_rate, precision, position, faulty_layers, checkpoint_path):

  in_channels = 3
  prec = 8 # Currently fault injection supported only for 8 bit quantization

  """ unperturbed model 
  """
  if arch == 'vgg11':
    model  = vggf('A',in_channels, 10, True, prec, 0,0, 0,0,[])
  elif arch == 'vgg16':
    model  = vggf('D',in_channels, 10, True, prec, 0, 0 ,0,0,[])
  elif arch == 'resnet18':
    model = resnetf('resnet18', 10, prec, 0,0,0,0,[]) 
  elif arch == 'resnet34':
    model = resnetf('resnet34', 10, prec, 0, 0,0,0,[]) 
  else:
    model = lenetf(in_channels,10,prec, 0,0,0,0,[])
  
  """ Perturbed model, where the weights are injected with bit errors at the rate of ber
  """
  rf = randomfault.RandomFaultModel(bit_error_rate, prec, position, 0)
  BitErrorMap0 = torch.tensor(rf.BitErrorMap_flip0).to(torch.int32).to(device)
  BitErrorMap1 = torch.tensor(rf.BitErrorMap_flip1).to(torch.int32).to(device)
  if arch == 'vgg11':
    model_p  = vggf('A',in_channels, 10, True, prec, bit_error_rate, position, BitErrorMap0, BitErrorMap1, faulty_layers)
  elif arch == 'vgg16':
    model_p  = vggf('D',in_channels, 10, True, prec, bit_error_rate, position, BitErrorMap0, BitErrorMap1, faulty_layers)
  elif arch == 'resnet18':
    model_p = resnetf('resnet18', 10, prec, bit_error_rate, position, BitErrorMap0, BitErrorMap1, faulty_layers) 
  elif arch == 'resnet34':
    model_p = resnetf('resnet34', 10, prec, bit_error_rate, position, BitErrorMap0, BitErrorMap1, faulty_layers) 
  else:
    model_p = lenetf(in_channels,10,prec, bit_error_rate, position, BitErrorMap0, BitErrorMap1, faulty_layers)
  
  print(model)
  print(model_p)

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


def transform_train(trainloader, arch, dataset, ber, precision, position, checkpoint_path, mean, std, device):
  
  #model = torch.nn.DataParallel(model)
  torch.backends.cudnn.benchmark = True


  model, model_perturbed = init_models(arch, ber, precision, position, cfg.faulty_layers, checkpoint_path)

  mean = mean[..., np.newaxis, np.newaxis]
  std = std[..., np.newaxis, np.newaxis]
  Pg = Program(cfg, model, model_perturbed, mean, std)
  criterion = nn.CrossEntropyLoss()
  #self.BCE = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Pg.parameters()),lr=cfg.learning_rate, betas=(0.5, 0.999), weight_decay=5e-4)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=cfg.decay)
  lb = cfg.lb
  for x in range(cfg.epochs):
    for batch_id, (image, label) in enumerate(trainloader):
      lr_scheduler.step()
      #image = tensor2var(image)
      image = image.to(device)
      label = label.to(device)
      out, out_biterror = Pg(image)
      loss_orig = compute_loss(out, label)
      loss_p = compute_loss(out_biterror, label)
      loss = loss_orig + lb*loss_p 
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    print('loss %.6f %.6f' %(loss_orig.cpu().numpy(), loss_p.cpu().numpy()))
    torch.save({'W': get_W}, '%s/W_%03d.pt' % (cfg.save_dir, x))
      #self.validate()
    if x%20==19:
      lb *= 0.5
      print("Lambda value: ", lb)



