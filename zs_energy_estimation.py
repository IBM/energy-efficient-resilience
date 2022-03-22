from __future__ import division

import numpy as np
import os 
import sys
import argparse
import pdb
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import zs_hooks_stats as stats

verbose=True


sys.path.append('./models')
from models import lenetf
from models import vggf
from models import resnetf
from config import cfg



layer_counter = 0
#batch_size = 10
#input_shape = (3, 32, 32)
layer_density = {}
energy_skip_comp = {}
energy_skip_inst = {}
energy_total = {}

class EnergyEstimation():

    def __init__(self):
        self.xvf32ger_energy = 395.2*0.505  # dynamic energy per instruction = 1 unit as of now since I need to check if this value can be shared.
        self.xvf32ger_multiplies = 16
        self.accumulators = 8 # 64B each
        self.xvf32ger_n = 4 # 4 instructions to create a 4x4 outer product of two 4-element vectors
        self.bblock_r = 8
        self.bblock_c = 16
        self.xvf32ger_energy_density_scale = [0.203665988, 0.476610249, 0.601375176, 0.700340856, 0.801657264, 0.865362012, 0.910437236, 0.963622473, 0.990185708, 0.998883404, 1]

    
    def baseline_energy_dataswitching(self, weight_size, input_size, output_size, density):
        """
        This estimates energy for a given layer, if there is no explicit support for sparsity. 0-valued computations are not eliminated. 
        The objective is to include the effect of data switching. This will be the baseline case.  
        """
        bblock8x16_xvf32ger_n = self.accumulators * self.xvf32ger_n # number of instructions to compute 8x8 block with 8x4 and 4x16 inputs 
        sparse_index = np.array(np.round(density*10), dtype=np.uint8)
        bblock8x16_energy = bblock8x16_xvf32ger_n * self.xvf32ger_energy * self.xvf32ger_energy_density_scale[sparse_index]
        """
         Matrix multiply compute MxN = MxK * KxN
         Each MxN output block is divided into computation blocks of size 8x16. 
         Outer product matrix multiply is computed for inputs of size 8xK, Kx16
        """
        M = output_size[0]
        K = input_size[0] # or weight_size[1]
        N = output_size[1]
        block8x16_xvf32ger_n = bblock8x16_xvf32ger_n * K/4
        block8x16_energy = K/4 * bblock8x16_energy
        
        block8x16_n = np.ceil(M/self.bblock_r) * np.ceil(N/self.bblock_c)
        energy = block8x16_n * block8x16_energy
        print('uniform sparse with data switching energy reduction', energy)
        return energy

    def uniform_sparsity_energy(self, weight_size, input_size, output_size, density):
        """
        This estimates energy for a given layer, if every basic block in that layer had the same activation density. 
        We assume that there is explicit support for fine-grained sparsity exploitation, by way of perfect run-time prediction of 0-valued inputs and fine-grained clock-gating.
        We assume that since the prediction capability exists, 0-valued computations can be eliminated from each of the 4x4 basic blocks, resulting in dynamic energy reduction .         
        """
        bblock8x16_xvf32ger_n = self.accumulators * self.xvf32ger_n 
        # number of instructions to compute 8x8 block with 8x4 and 4x16 inputs 
        computations_reduced = np.floor((1-density) * 4) * 4
        # Each 0-valued activation in 4-element vector results in elimination of 4 computations

        energy_reduction = (self.xvf32ger_energy/self.xvf32ger_multiplies)* computations_reduced 
        sparse_energy = self.xvf32ger_energy - energy_reduction
        bblock8x16_energy = bblock8x16_xvf32ger_n * sparse_energy 

        M = output_size[0]
        K = input_size[0] # or weight_size[1]
        N = output_size[1]
        block8x16_xvf32ger_n = bblock8x16_xvf32ger_n * K/4
        block8x16_energy = K/4 * bblock8x16_energy
        
        block8x16_n = np.ceil(M/self.bblock_r) * np.ceil(N/self.bblock_c)
        energy = block8x16_n * block8x16_energy
        print('uniform sparse with 0-valued computes eliminated', energy)
        return energy

    def sparse_energy(self, activations, weights, input_size, weight_size, output_size):
        # Here activations is the 0-padded matrix
        ishape = list(activations.shape)
        wshape = list(weights.shape)

        xvf32ger_instructions_eliminated = 0
        xvf32ger_instructions = 0
        xvf32ger_energy_total = 0.0
        xvf32ger_energy_skip_inst = 0.0
        xvf32ger_energy_skip_comp = 0.0
        for b in range(0,ishape[0]): # batches
            for c in range(0,wshape[1]): # input channels
    
                #print(activations[b,c,:,:])
                for i in range(0,wshape[2]): # row stride      
                    for j in range(0,wshape[3]): # column stride

                        # assemble elements of one row of MMA activation matrix
                        rowCount = 0 
                        for k in range(i,ishape[2]-wshape[2]+i+1): # rows of activations 
                            #print(b,c,k,j, ishape[3]-wshape[3]+j+1) 
                            if rowCount==0: 
                                mma_act_row = activations[b,c,k,j:ishape[3]-wshape[3]+j+1] # choose elements
                            else: 
                                mma_act_row = torch.cat((mma_act_row,activations[b,c,k,j:ishape[3]-wshape[3]+j+1]),0) 
                            rowCount += 1
                        #print(mma_act_row) 
                        #pdb.set_trace()

                        # check for density in each 4-element sequence
                        mshape=list(mma_act_row.shape)
                        for m in range(0,np.ceil(mshape[0]/4).astype(int)):
                            xvf32ger_instructions += 1

                            density = 1 - (torch.sum(mma_act_row.view(-1)[:4:] == 0))/4.0
                            if (density == 0):
                                xvf32ger_instructions_eliminated += 1
                            else:
                                sparse_index = (density*10).to(torch.uint8)
                                xvf32ger_energy_skip_inst += self.xvf32ger_energy_density_scale[sparse_index] * self.xvf32ger_energy

                            computes_skipped = torch.sum(mma_act_row.view(-1)[:4:] == 0)
                            energy_reduction = (self.xvf32ger_energy/self.xvf32ger_multiplies)* computes_skipped * 4 # each zero-valued activation results in 4 0-valued partial products among 16 
                            sparse_energy = self.xvf32ger_energy - energy_reduction
                            xvf32ger_energy_skip_comp += sparse_energy

                            # baseline energy without skipping sparse computations or instructions 
                            sparse_index = (density*10).to(torch.uint8)
                            xvf32ger_energy_total += self.xvf32ger_energy_density_scale[sparse_index] * self.xvf32ger_energy

                            mma_act_row = mma_act_row.view(-1)[4::]
 
        M = weight_size[0]
        K = input_size[0]
        N = input_size[1]
        xvf32ger_instructions_eliminated = xvf32ger_instructions_eliminated * (M/4)                 
        xvf32ger_instructions = xvf32ger_instructions * (M/4)
        xvf32ger_energy_total = xvf32ger_energy_total * (M/4)
        xvf32ger_energy_skip_inst = xvf32ger_energy_skip_inst * (M/4)
        xvf32ger_energy_skip_comp = xvf32ger_energy_skip_comp * (M/4)

        if (verbose):
            print(M,K,N)
            print('percentage instructions eliminated %.4f'%(xvf32ger_instructions_eliminated/xvf32ger_instructions))
            print('Total energy baseline %.4f'%(xvf32ger_energy_total))
            print('Total energy with skipped instructions %.4f'%(xvf32ger_energy_skip_inst))
            print('Total energy with skipped computations per instruction %.4f'%(xvf32ger_energy_skip_comp))

        ## verify code to compute # instructions
        #bblock8x16_xvf32ger_n = self.accumulators * self.xvf32ger_n  
        #block8x16_xvf32ger_n = bblock8x16_xvf32ger_n * K/4
        #block8x16_n = np.ceil(M/self.bblock_r) * np.ceil(N/self.bblock_c)
        #print('total number of instructions', block8x16_n*block8x16_xvf32ger_n)                            
        
        # verify energy reduction    
        # print('Energy reduced %.4f'%(xvf32ger_instructions_eliminated*self.xvf32ger_energy_density_scale[0]*self.xvf32ger_energy))

        return xvf32ger_energy_total, xvf32ger_energy_skip_inst, xvf32ger_energy_skip_comp


def mma_instructions_estimate(conv2d_i, conv2d_w, conv2d_o):

    with torch.no_grad():
        conv2d_input = conv2d_i.clone().detach()
        conv2d_weight = conv2d_w.clone().detach()
        conv2d_output = conv2d_o.clone().detach()

        ee = EnergyEstimation()

        i_shape = list(conv2d_input.shape)
        o_shape = list(conv2d_output.shape)
        w_shape = list(conv2d_weight.shape)
        print('input', i_shape, 'output', o_shape, 'weight', w_shape)

        # deriving matrix shapes for MMA instructions

        mma_weight_shape = [w_shape[0], w_shape[1]*w_shape[2]*w_shape[3]]
        mma_input_shape = [w_shape[1]*w_shape[2]*w_shape[3], o_shape[2]*o_shape[3]]
        mma_output_shape = [o_shape[1], o_shape[2]*o_shape[3]] 
        
        print(mma_weight_shape, mma_input_shape, mma_output_shape)

        # 0 sparsity in data (just for verification)
        #ee.no_sparsity_energy(mma_weight_shape, mma_input_shape, mma_output_shape)

        # per-layer sparsity, where the computations that can be skipped are randomly distributed, and the % density of activations is the same for each block
        #density = torch.count_nonzero(conv2d_input)/(i_shape[0]*i_shape[1]*i_shape[2]*i_shape[3])
        density = 1 - (torch.sum(conv2d_input==0)*1.0/(i_shape[0]*i_shape[1]*i_shape[2]*i_shape[3]))
        print('layer density %.4f'%(density))
        #ee.uniform_sparsity_energy(mma_weight_shape, mma_input_shape, mma_output_shape, density.cpu().numpy())

        # zeropadded activation matrix to compute instruction level sparsity
        zpad = nn.ZeroPad2d(w_shape[2]-2)
        conv2d_input_z = zpad(conv2d_input)
        iz_shape = list(conv2d_input_z.shape)
        xvf32ger_energy_total, xvf32ger_energy_skip_inst, xvf32ger_energy_skip_comp = ee.sparse_energy(conv2d_input_z, conv2d_weight, mma_input_shape, mma_weight_shape, mma_output_shape)
        
        return density, xvf32ger_energy_total, xvf32ger_energy_skip_inst, xvf32ger_energy_skip_comp
                



def activations(self, input, output):
    global layer_counter, energy_skip_inst, energy_skip_comp
    if 'Conv2d' in self.__class__.__name__:
        density, xvf32ger_energy_total, xvf32ger_energy_skip_inst, xvf32ger_energy_skip_comp = mma_instructions_estimate(input[0], self.weight, output)
        layer_density[layer_counter] = density
        energy_total[layer_counter] = xvf32ger_energy_total
        energy_skip_inst[layer_counter] = xvf32ger_energy_skip_inst
        energy_skip_comp[layer_counter] = xvf32ger_energy_skip_comp

    layer_counter += 1


def init_models(arch, precision, checkpoint_path, device):

    in_channels = 3

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
    #print(model)

    model = model.to(device)

    print('Restoring model from checkpoint', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    print('restored checkpoint at epoch - ',checkpoint['epoch'])
    print('Training loss =', checkpoint['loss'])
    print('Training accuracy =', checkpoint['accuracy'])
    checkpoint_epoch=checkpoint['epoch']


    return model

def inference_energy(testloader, arch, dataset, precision, checkpoint_path, device):


    model = init_models(arch, precision, checkpoint_path, device)

    logger = stats.DataLogger(int(len(testloader.dataset)/testloader.batch_size), testloader.batch_size)

    hooks = {}
    for name, module in model.named_modules():
        module.module_name = name
        hooks[name] = module.register_forward_hook(activations)

    

    model.eval()

    model=model.to(device)

    with torch.no_grad():
        for t, (inputs,classes) in enumerate(testloader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            model_outputs =model(inputs)
            #pdb.set_trace()
            lg, preds = torch.max(model_outputs, 1)
            correct=torch.sum(preds == classes.data)
            
            logger.update(model_outputs)

    #logger.visualize()
    # forward pass of image perturbed with the program
    f = open("outputs.txt","w")
    f.write(str(layer_density))
    f.write(str(energy_total))
    f.write(str(energy_skip_inst))
    f.write(str(energy_skip_comp))
    f.close()
    
