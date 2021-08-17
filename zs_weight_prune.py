import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import re
import pdb
class FooBarPruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask

def foobar_unstructured(module, name):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
        >>> m = nn.Linear(3, 4)
        >>> foobar_unstructured(m, name='bias')
    """
    FooBarPruningMethod.apply(module, name)
    
    return module


def pruneweights(trainloader, model, arch, dataset, precision, checkpoint_path, device):
    print('Restoring model from checkpoint', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    print('restored checkpoint at epoch - ',checkpoint['epoch'])
    print('Training loss =', checkpoint['loss'])
    print('Training accuracy =', checkpoint['accuracy'])
    checkpoint_epoch=checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']

    for name, param in model.named_parameters():
        if "weight" in name: 
            print(param.size())
            #pdb.set_trace()
            print(float(torch.sum(param == 0))/float(param.nelement()))
    print((model.conv1.weight[0,0,:,:]))
    for name, module in enumerate(model.named_modules()):
        print(name)

        #if isinstance(module, torch.nn.Conv2d):
        #    print('pruning')
    foobar_unstructured(model.conv1, name="weight")
    #prune.l1_unstructured(model.conv1, name='weight', amount=0.3)
    print((model.conv1.weight[0,0,:,:]))
    #for name, param in model.named_parameters():
    #    if "weight" in name: 
    #        print(param.size())
    #        print(float(torch.sum(param == 0))/float(param.nelement()))

    model_path = arch + '_' + dataset  + '_p_'+ str(precision) + '_pruned_model_' + str(checkpoint_epoch)+ '.pth'
    torch.save({'epoch': (checkpoint_epoch), 'model_state_dict': model.state_dict(), 'loss': loss, 'accuracy': accuracy}, model_path)


def train(trainloader, model, arch, dataset, learning_rate, epochs, voltage, ber, precision, position, error_type, retrain, checkpoint_path, faulty_layers, device ):


    print(model)
    print('Training with Learning rate %.4f'%(learning_rate))
    opt = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)

    model = model.to(device)
    #model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    
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
        if x == 120:
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
        
           
            



