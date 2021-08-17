
import numpy as np
import os 
import sys
import argparse
#from gen_faultmap import *
#from defs import *
import pdb
import realm_utils as utils
import torch
import matplotlib.pyplot as plt
import multiprocessing as mp

visualize=False

## This class of model is characterized by the following parameters - 
# ber - bit error rate = count of faulty bits / total bits at a given voltage
# prob - likelihood of a faulty bit cell being faulty -- i.e likelihood of a bit cell being faulty on repeated access -- is it a transient fault ?
# ber0 - fraction of faulty bit cells that default to 0. (ber1 = ber - ber0)
# Assume each bit is likely to be faulty; ie sample from a uniform distribution to generate a spatial distribution of faults

class RandomFaultModels:
    MEM_ROWS=8192
    MEM_COLS=128
    prob=1.0  ## temporal likelihood of a given bit failing for a given access 
    ber0 = 0.5
#   BitErrorRate = [0.01212883, 0.00397706, 0.001214473, 0.00015521, 0.000126225, 4.06934E-05, 1.3119E-05] # Count of faulty bits / total bits for 7 operating points. 

    def __init__(self, ber, prec, pos, seed):
        self.ber = ber
        self.ber0 = RandomFaultModels.ber0
        self.precision = prec
        self.MEM_ROWS = RandomFaultModels.MEM_ROWS
        self.MEM_COLS = RandomFaultModels.MEM_COLS
        print('Bit Error Rate %.3f Precision %d Position %d'%(self.ber, self.precision, pos))
        if pos==-1:
            #self.BitErrorMap_flip0, self.BitErrorMap_flip1 = self.ReadBitErrorMap()
            self.BitErrorMap_flip0, self.BitErrorMap_flip1 = self.GenBitErrorMap(seed)
        else:
            self.BitErrorMap_flip0, self.BitErrorMap_flip1 = self.GenBitPositionErrorMap(pos)

#    ## for debug only ##
#    def ReadBitErrorMap(self):
#        mem_voltage = self.voltage
#        chip = 'n'
#        fname = './faultmaps_chip_'+ chip + '/fmap_sa0_v_' + str(mem_voltage) + '.txt'
#        some_arr=np.genfromtxt(fname, dtype='uint32', delimiter=',')
#        bitmap_flip0=some_arr[0:self.MEM_ROWS, 0:self.MEM_COLS] 
#        ## tiling it at the moment 
#        print('SA 0 Bit error rate', (bitmap_flip0.sum()/(self.MEM_ROWS*self.MEM_COLS)))
#        fname = './faultmaps_chip_' + chip + '/fmap_sa1_v_' + str(mem_voltage) + '.txt'
#        some_arr=np.genfromtxt(fname, dtype='uint32', delimiter=',')
#        bitmap_flip1=some_arr[0:self.MEM_ROWS, 0:self.MEM_COLS] 
#        print('SA 1 Bit error rate', (bitmap_flip1.sum()/(self.MEM_ROWS*self.MEM_COLS)))
#        return bitmap_flip0, bitmap_flip1
#
    def GenBitErrorMap(self, seed):

        

        bitmap= np.zeros((self.MEM_ROWS,self.MEM_COLS))
        bitmap_flip0= np.zeros((self.MEM_ROWS,self.MEM_COLS))
        bitmap_flip1= np.zeros((self.MEM_ROWS,self.MEM_COLS))

        if (seed != None):
            np.random.seed(seed)
        bitmap_t = np.random.rand(self.MEM_ROWS,self.MEM_COLS)
        bitmap[bitmap_t < self.ber] = 1       

        #print(bitmap)
        if (seed != None):
            np.random.seed(seed+1)
        bitmap_flip = np.random.rand(self.MEM_ROWS,self.MEM_COLS)

        bitmap_flip0[bitmap_flip < self.ber0] = 1
        bitmap_flip1[bitmap_flip >= self.ber0] = 1
        #print(bitmap_flip0)
        #print(bitmap_flip1)
        bitmap_flip0 = bitmap*bitmap_flip0
        bitmap_flip1 = bitmap*bitmap_flip1
    
        #print(bitmap_flip0)
        #print(bitmap_flip1)
        bitmap_flip0 = bitmap_flip0.astype(np.uint32)
        bitmap_flip1 = bitmap_flip1.astype(np.uint32)
        bitcells = self.MEM_ROWS*self.MEM_COLS
        print('Read 0 Bit Error Rate', sum(sum(bitmap_flip0))/bitcells)
        print('Read 1 Bit Error Rate', sum(sum(bitmap_flip1))/bitcells)
        return bitmap_flip0, bitmap_flip1

    def GenBitPositionErrorMap(self, pos):

        bitmap= np.zeros((self.MEM_ROWS,self.MEM_COLS))
        bitmap_flip0= np.zeros((self.MEM_ROWS,self.MEM_COLS))
        bitmap_flip1= np.zeros((self.MEM_ROWS,self.MEM_COLS))

        # Generate errors at rate ber in a specific bit position, maximum of one error per weight in the specified position
        weights_per_row = int(self.MEM_COLS/self.precision)
        bitmap_pos = np.zeros((self.MEM_ROWS,weights_per_row))
        bitmap_t = np.random.rand(self.MEM_ROWS,weights_per_row)
        bitmap_pos[bitmap_t < self.ber] = 1       
        # Insert the faulty column in bit error map
        for k in range(0,weights_per_row):
            bitmap[:, k*self.precision+pos] = bitmap_pos[:,k]
        #print(bitmap)

        bitmap_flip = np.random.rand(self.MEM_ROWS,self.MEM_COLS)
        bitmap_flip0[bitmap_flip < self.ber0] = 1
        bitmap_flip1[bitmap_flip >= self.ber0] = 1
        #print(bitmap_flip0)
        #print(bitmap_flip1)
        bitmap_flip0 = bitmap*bitmap_flip0
        bitmap_flip1 = bitmap*bitmap_flip1
    
        #print(bitmap_flip0)
        #print(bitmap_flip1)
        bitmap_flip0 = bitmap_flip0.astype(np.uint32)
        bitmap_flip1 = bitmap_flip1.astype(np.uint32)
        bitcells = self.MEM_ROWS*self.MEM_COLS
        print('Bit Error Rate', sum(sum(bitmap_flip0))/bitcells + sum(sum(bitmap_flip1))/bitcells)
        return bitmap_flip0, bitmap_flip1

    def ConvertToMemoryLayout(self,bmap,rows,cols,numBanks):
        arr=np.tile(bmap, [numBanks, 1])

        BitErrorMap_t = np.zeros([rows,cols],dtype=np.uint32)

        for k in range(0,cols):
            for j in range(0,self.precision):
                BitErrorMap_t[:,k] = (BitErrorMap_t[:,k]) + np.left_shift(arr[:,k*self.precision+j],j)
        
        if self.precision == 8: 
            BitErrorMap = np.array(np.reshape(BitErrorMap_t, [rows*cols,1]), dtype=np.uint8)
        elif self.precision == 16:
            BitErrorMap = np.array(np.reshape(BitErrorMap_t, [rows*cols,1]), dtype=np.uint16)
        else:
            BitErrorMap = np.array(np.reshape(BitErrorMap_t, [rows*cols,1]), dtype=np.uint32)
        return BitErrorMap

    def MapWeightsToBitErrors(self, numWeights):
        # size of the binary array that holds the faultmap, each row has fault map for 1 weight
        MAX_WEIGHTS_PER_BANK = (self.MEM_ROWS*self.MEM_COLS/self.precision)
        numBanks = int(np.ceil(numWeights/MAX_WEIGHTS_PER_BANK))
        cols = int(self.MEM_COLS/self.precision)
        rows = self.MEM_ROWS*numBanks
        BitErrorMap0 = ~self.ConvertToMemoryLayout(self.BitErrorMap_flip0,rows,cols,numBanks)
        BitErrorMap1 = self.ConvertToMemoryLayout(self.BitErrorMap_flip1,rows,cols,numBanks)
        if self.precision == 8: 
            BitErrorMap0 = BitErrorMap0.astype(np.int8)
            BitErrorMap1 = BitErrorMap1.astype(np.int8)
        elif self.precision == 16:
            BitErrorMap0 = BitErrorMap0.astype(np.int16)
            BitErrorMap1 = BitErrorMap1.astype(np.int16)
        else:
            BitErrorMap0 = BitErrorMap0.astype(np.int32)
            BitErrorMap1 = BitErrorMap1.astype(np.int32)
            
        return BitErrorMap0[0:numWeights,:],  BitErrorMap1[0:numWeights,:]


    def InjectWeights(self, weights):

        #Inject faults into weights
        numWeights=np.prod(weights.shape)
        weights_r = np.reshape(weights,[numWeights,1])
        weights_int,delta = utils.quantize(weights_r, self.precision)
        if (visualize == True):
            utils.collect_hist(weights_r)

        BitErrorMap0, BitErrorMap1 = self.MapWeightsToBitErrors(numWeights)
        
        weights_a = np.bitwise_and(BitErrorMap0,weights_int)
        weights_f = np.bitwise_or(BitErrorMap1,weights_a)
        weights_faulty = utils.dequantize(weights_f,delta) 
        #print(np.max(abs(weights_r)), np.max(weights_r), np.min(weights_r))
        #print(np.max(abs(weights_faulty)), np.max(weights_faulty), np.min(weights_faulty))
        if (visualize == True):
            utils.collect_hist(weights_faulty, weights_r)
        d2,di,d0 = utils.collect_norms(weights_faulty,weights_r)
        #print(d2,di,d0)
        weights_faulty = np.reshape(weights_faulty,weights.shape)    
        return weights_faulty, d2, di, d0

### end class 

def FaultInject_func(params,ind):
	
    #print('faulty layer', ind)
    #cpu_weights = params[x].to("cpu")
    cpu_weights = params[ind].cpu()
    weights = cpu_weights.detach().numpy()
    weights_faulty, d2, di, d0 = RFM.InjectWeights(weights)
#    l0_norm[0,x] = d0
#    l2_norm[0,x] = d2
#    li_norm[0,x] = di


    weights_f = torch.Tensor(weights_faulty)
    weights_f = weights_f.to(device)

    ## Works 
    params[ind].data = weights_f.data

    #params[ind].copy_(weights_f)  ## In place operation Not allowed on a leaf node that is part of the graph

    ## Does not work -- modification not reflected in actual model params
    #with torch.no_grad():
    #    params[ind] = weights_f

def faulty_weights(params, faulty_layers, ber, precision, position, device, seed, RFM):

    if (seed== None):
        RFM = RandomFaultModels(ber,precision,position,None)

    l0_norm = np.zeros((1, len(faulty_layers)))
    l2_norm = np.zeros((1, len(faulty_layers)))
    li_norm = np.zeros((1, len(faulty_layers)))
    for x in range(len(faulty_layers)):

        ind = faulty_layers[x]
        p = mp.Process(target=FaultInject_func, args=(params,ind,))
        processes.append(p)
        p.start()


    for process in processes:
	    process.join()

    return l0_norm,l2_norm,li_norm



####  Test code 
#def faulty_weights():
#    #np.random.seed(1)
#    num=1000
#    ber=0.3
#    weights = np.random.randn(num,num)
#    RFM = RandomFaultModels(ber)
#
#    #l0_norm = np.zeros((1, len(faulty_layers)))
#    #l2_norm = np.zeros((1, len(faulty_layers)))
#    #li_norm = np.zeros((1, len(faulty_layers)))
#    #for x in range(len(faulty_layers)):
#    #    ind = faulty_layers[x]
#        #cpu_weights = params[x].to("cpu")
#    #    cpu_weights = params[ind].cpu()
#    #    weights = cpu_weights.detach().numpy()
#        #pdb.set_trace()
#        #print(np.max(weights))
#        #print(weights.dtype) ## float32
#        
#    weights_faulty, d2, di, d0 = RFM.InjectWeights(weights)
#
#
#
#if __name__ == "__main__":
#    faulty_weights()

