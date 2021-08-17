
import numpy as np
import os 
import sys
import argparse
import pdb
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

debug=False


torch.manual_seed(0)
def training(trainloader, model, arch, dataset, learning_rate, epochs, voltage, ber, precision, position, error_type, retrain, checkpoint_path, faulty_layers, device ):


    print(model)
    print('Training with Learning rate %.4f'%(learning_rate))
    opt = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)

    model = model.to(device)
    #model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    checkpoint_epoch=0
    if (retrain):
        print('Restoring model from checkpoint', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        print('restored checkpoint at epoch - ',checkpoint['epoch'])
        print('Training loss =', checkpoint['loss'])
        print('Training accuracy =', checkpoint['accuracy'])
        checkpoint_epoch=checkpoint['epoch']

    curr_lr=learning_rate
    for x in range(epochs):

        running_loss = 0.0
        running_correct = 0
        for batch_id, (inputs,outputs) in enumerate(trainloader):
            inputs = inputs.to(device)
            outputs = outputs.to(device)
        
            opt.zero_grad()
            
            # Store original model parameters before quantization/perturbation, detached from graph
            if(precision > 0): 
                list_init_params = []
                with torch.no_grad():
                    for init_params in model.parameters():
                        list_init_params.append(init_params.clone().detach())

                if (debug):
                    if (batch_id % 100 == 0):
                        print('initial params')
                        print(model.fc2.weight[0:3,0:3])
                        print(model.conv1.weight[0,0,:,:])
                

            model_outputs = model(inputs)

            _, preds = torch.max(model_outputs, 1)
            outputs = outputs.view(outputs.size(0))  ## changing the size from (batch_size,1) to batch_size. 

            if (precision > 0):
                if (debug):
                    if (batch_id % 100 == 0):
                        print('quantized params')
                        print(model.fc2.weight[0:3,0:3])
                        print(model.conv1.weight[0,0,:,:])

            loss = nn.CrossEntropyLoss()(model_outputs, outputs)

            if (debug):
                print('epoch %d batch %d loss %.3f'%(x,batch_id,loss))

            # Compute gradient of perturbed weights with perturbed loss 
            loss.backward()

            # restore model weights with unquantized value
            if (precision > 0):
                with torch.no_grad():
                    for i,restored_params in enumerate(model.parameters()):
                        restored_params.copy_(list_init_params[i])

                if (debug):
                    if (batch_id % 100 == 0):
                        print('restored params')
                        print(model.fc2.weight[0:3,0:3])
                        print(model.conv1.weight[0,0,:,:])

            # update restored weights with gradient 
            opt.step()


            running_loss+=loss.item()
            running_correct+=torch.sum(preds == outputs.data)

        # update learning rate
        if ((x==80) or (x == 120)):
            curr_lr /= 10.0
            for param_group in opt.param_groups:
                param_group['lr'] = curr_lr
            print('Training with Learning rate %.4f'%(curr_lr))
     
            
        accuracy = running_correct.double()/(len(trainloader.dataset))
        print('epoch %d loss %.6f accuracy %.6f' %(x, running_loss/(batch_id), accuracy))
        #writer.add_scalar('Loss/train', running_loss/batch_id, x)   ## loss/#batches 
        if ((x)%40 == 0) or (x==epochs-1):
            model_path = arch + '_' + dataset  + '_p_'+ str(precision) + '_model_' + str(checkpoint_epoch+x)+ '.pth'
            torch.save({'epoch': (checkpoint_epoch+x), 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'loss': running_loss/batch_id, 'accuracy': accuracy}, model_path)
                #utils.collect_gradients(params, faulty_layers)
        
           
            



