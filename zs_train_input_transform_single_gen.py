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
import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy
import tqdm

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


class Generator(nn.Module):
    """
    Apply reprogramming.
    """
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg
        self.num_classes = 10

        # Encoder
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # torch.nn.init.xavier_uniform(self.conv1_1.weight)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # torch.nn.init.xavier_uniform(self.conv2_1.weight)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # torch.nn.init.xavier_uniform(self.conv3_1.weight)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.relu3_1 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.dconv4_1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        # torch.nn.init.xavier_uniform(self.dconv4_1.weight)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.relu4_1 = nn.ReLU()

        self.dconv5_1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        # torch.nn.init.xavier_uniform(self.dconv5_1.weight)
        self.bn5_1 = nn.BatchNorm2d(32)
        self.relu5_1 = nn.ReLU()

        self.dconv6_1 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1)
        # torch.nn.init.xavier_uniform(self.dconv6_1.weight)
        self.bn6_1 = nn.BatchNorm2d(3)
        self.tanh = torch.nn.Tanh()

    def forward(self, image):
        img = image.data.clone()
        # Encoder
        x = self.relu1_1(self.bn1_1(self.conv1_1(img)))
        x = self.maxpool1(x)
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.maxpool2(x)
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.maxpool3(x)

        # Decoder
        x = self.relu4_1(self.bn4_1(self.dconv4_1(x)))
        x = self.relu5_1(self.bn5_1(self.dconv5_1(x)))
        x = self.bn6_1(self.dconv6_1(x))
        out = self.tanh(x)

        x_adv = torch.clamp(image + out, min=-1, max=1)

        return x_adv, out


def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(
        labels.size(0)
    )  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds

def compute_diversity_loss(img_origin, img_random, gen, device):
    """
    Calculate the diversity.
    :param img_origin: The image without apply input transformation.
    :param img_random: The image without apply input transformation, and it is sample from the DataLoader randomly and img_random != img_origin.
    :param gen: Generator object to generate the perturbation based on the input images.
    :param device: Specify GPU usage.
    """
    diversityLoss = 0
    perturb_origin, _ = gen(img_origin) # Generate transformed images.
    perturb_random, _ = gen(img_random) # Generate transformed images.
    perturb_origin = perturb_origin.to(device)
    perturb_random = perturb_random.to(device)
    
    # Calculate the reverse Lipschitz continous loss.
    for img_o, img_r, pertrub_o, perturb_r in zip(img_origin, img_random, perturb_origin, perturb_random):
        diff_img_norm = torch.linalg.norm(torch.reshape(img_o-img_r, (-1,) ), dim=0, ord=2)
        diff_p_norm = torch.linalg.norm(torch.reshape(pertrub_o-perturb_r, (-1,) ), dim=0, ord=2)
        diversityLoss += diff_img_norm / diff_p_norm
    return diversityLoss
                

def accuracy_checking(model_orig, model_p, trainloader, testloader, gen, device, use_transform=False):
    """
    Check the accuracy for both training data and testing data.
    :param model_orig: The clean model.
    :param model_p: The perturbed model.
    :param trainloader: The loader of training data.
    :param testloader: The loader of testing data.
    :param gen: Generator object to generate the perturbation based on the input images.
    :param device: Specify GPU usage.
    :param use_transform: Whether to apply input transfomation or not.
    """
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
            x_adv, _ = gen(x)
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
            x_adv, _ = gen(x)
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

# For hooking the activation from clean model and clean images.
def get_activation_c(act_c, name):
    def hook(model, input, output):
        act_c[name] = output
    return hook

# For hooking the activation from perturbed model and transformed images.
def get_activation_p(act_p, name):
    def hook(model, input, output):
        act_p[name] = output
    return hook

def calActivationDiff(actCdict, actPdict, actdict, act_c, act_p):
    """
    Calculate the difference of the activations from actCdict and actPdict.
    :param actCdict: Dict. The activation from clean model and clean images.
    :param actPdict: Dict. The activation from perturbed model and transformed images.
    :param actdict: List. Store the 2-norm of the difference between actCdict and actPdict.
    :param act_c: List. Store the 2-norm of the activation from actCdict.
    :param act_p: List. Store the 2-norm of the activation from actPdict.
    """
    layer_keys = act_c.keys()
    for name in layer_keys:
        #print(act_c[name].shape)
        cleanAct = torch.linalg.norm(torch.reshape(act_c[name], (act_c[name].shape[0], -1)), dim=1, ord=2)
        perturbedAct = torch.linalg.norm(torch.reshape(act_p[name], (act_p[name].shape[0], -1)), dim=1, ord=2)
        diff_norm = torch.linalg.norm(torch.reshape(act_c[name]-act_p[name], (act_c[name].shape[0], -1)), dim=1, ord=2)
        # print(torch.abs(cleanAct-perturbedAct).shape)
        actdict[name].append(diff_norm.data.cpu())
        actCdict[name].append(cleanAct.data.cpu())
        actPdict[name].append(perturbedAct.data.cpu())
        # print(diff.data)

def calOutputDiff(outClean, outPerturbed, outDiff, outputNp, outputP):
    """
    Calculate the difference of the output between outClean and outPerturbed.
    :param actCdict: Dict. The activation from clean model and clean images.
    :param actPdict: Dict. The activation from perturbed model and transformed images.
    :param actdict: List. Store the 2-norm of the difference between actCdict and actPdict.
    :param act_c: List. Store the 2-norm of the activation from actCdict.
    :param act_p: List. Store the 2-norm of the activation from actPdict.
    """

    #print(act_c[name].shape)
    cleanOut = torch.linalg.norm(torch.reshape(outputNp, (outputNp.shape[0], -1)), dim=1, ord=2)
    perturbedOut = torch.linalg.norm(torch.reshape(outputP, (outputP.shape[0], -1)), dim=1, ord=2)
    diff_norm = torch.linalg.norm(torch.reshape(outputNp-outputP, (outputNp.shape[0], -1)), dim=1, ord=2)
    # print(torch.abs(cleanAct-perturbedAct).shape)
    outDiff.append(diff_norm.data.cpu())
    outClean.append(cleanOut.data.cpu())
    outPerturbed.append(perturbedOut.data.cpu())
    # print(diff.data)

def layerwise(act_c, act_p):
    sumLoss = 0
    MSE = nn.MSELoss()
    layer_keys = act_c.keys()
    for name in layer_keys:
        #print(MSE(act_c[name], act_p[name]))
        sumLoss += MSE(act_c[name], act_p[name])
    return sumLoss

def checkActivationDistribution(model_np_origin, model_p, trainDataLoader, gen, act_c, act_p, act_diff, act_clean, act_perturbed, epoch):

    print('Calculating activation values')

    model_np_origin.eval()
    model_p.eval()
    gen.eval()
    for batch_id, (image, label) in tqdm.tqdm(enumerate(trainDataLoader)):
        image, label = image.to(device), label.to(device)
        image_adv, _ = gen(image)  # pylint: disable=E1102
        image_adv = image_adv.to(device)
        
        _, _ = model_np_origin(image), model_p(image_adv)
        calActivationDiff(act_clean, act_perturbed, act_diff, act_c, act_p) 

    for name, layer in model_np_origin.named_modules():
        if 'relu' in name:
            # Draw learning curve:
            # print(len(list(itertools.chain(*act_diff[name]))))
            reshapeActList = list(itertools.chain(*act_diff[name]))
            reshapeActList = [act.item() for act in reshapeActList]

            reshapeActCleanList = list(itertools.chain(*act_clean[name]))
            reshapeActCleanList = [act.item() for act in reshapeActCleanList]

            reshapeActPerturbedList = list(itertools.chain(*act_perturbed[name]))
            reshapeActPerturbedList = [act.item() for act in reshapeActPerturbedList]

            plt.hist(reshapeActList, color = 'green')
            # plt.hist(reshapeActList, bins=np.arange(min(reshapeActList), max(reshapeActList) + 10, 10), color = "skyblue", ec="skyblue")
            plt.title('L2 norm different between clean / perturbed model: {}'.format(name))
            plt.savefig(cfg.save_dir + 'NormDiff_Layer_{}_Epoch_{}.jpg'.format(name, epoch+1))
            plt.clf()

            plt.hist(reshapeActCleanList, color = 'blue')
            plt.hist(reshapeActPerturbedList, color = 'red')
            plt.legend(['Clean model', 'Perturbed model'])
            # plt.hist(reshapeActList, bins=np.arange(min(reshapeActList), max(reshapeActList) + 10, 10), color = "skyblue", ec="skyblue")
            plt.title('Activation distribution: {}'.format(name))
            plt.savefig(cfg.save_dir + 'Distribution_Layer_{}_Epoch_{}.jpg'.format(name, epoch+1))
            plt.clf()

def checkOutputDistribution(model_np_origin, model_p, trainDataLoader, gen, out_diff, out_clean, out_perturbed, epoch):

    print('Calculating output values')

    model_np_origin.eval()
    model_p.eval()
    gen.eval()
    for batch_id, (image, label) in tqdm.tqdm(enumerate(trainDataLoader)):
        image, label = image.to(device), label.to(device)
        image_adv = gen, _(image)  # pylint: disable=E1102
        image_adv = image_adv.to(device)
        
        output_np, output_p = model_np_origin(image), model_p(image_adv)
        calOutputDiff(out_clean, out_perturbed, out_diff, output_np, output_p) 


    # Draw learning curve:
    # print(len(list(itertools.chain(*act_diff[name]))))
    reshapeOutputList = list(itertools.chain(*out_diff))
    reshapeOutputList = [out.item() for out in reshapeOutputList]
    reshapeOutputCleanList = list(itertools.chain(*out_clean))
    reshapeOutputCleanList = [out.item() for out in reshapeOutputCleanList]
    reshapeOutputPerturbedList = list(itertools.chain(*out_perturbed))
    reshapeOutputPerturbedList = [out.item() for out in reshapeOutputPerturbedList]
    plt.hist(reshapeOutputList, color = 'green')
    # plt.hist(reshapeActList, bins=np.arange(min(reshapeActList), max(reshapeActList) + 10, 10), color = "skyblue", ec="skyblue")
    plt.title('L2 norm different between clean / perturbed model')
    plt.savefig(cfg.save_dir + 'OutputNormDiff_Epoch_{}.jpg'.format(epoch))
    plt.clf()

    plt.hist(reshapeOutputCleanList, color = 'blue')
    plt.hist(reshapeOutputPerturbedList, color = 'red')
    plt.legend(['Clean model', 'Perturbed model'])
    # plt.hist(reshapeActList, bins=np.arange(min(reshapeActList), max(reshapeActList) + 10, 10), color = "skyblue", ec="skyblue")
    plt.title('Output distribution')
    plt.savefig(cfg.save_dir + 'OutputDistribution_Epoch_{}.jpg'.format(epoch))
    plt.clf()

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

    print('Seed: {}'.format(seed))

    storeLoss = []

    tmpTrainLoader = copy.deepcopy(trainloader)

    model_np = copy.deepcopy(model) # Inference origin images

    Gen = Generator(cfg)
    Gen = Gen.to(device)
    Gen.train()

    from torchsummary import summary
    summary(Gen, (3, 32, 32))
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, Gen.parameters()),
        lr=cfg.learning_rate,
        betas=(0.5, 0.999),
        # weight_decay=5e-4,
    )

    '''
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, gen.parameters()),
        lr=cfg.learning_rate,
        momentum=0.9, 
        #weight_decay=5e-4
    )
    '''

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2000, gamma=cfg.decay
    )

    lb = cfg.lb  # Lambda

    for name, param in Gen.named_parameters():
        print("Param name: {}, grads is: {}".format(name, param.requires_grad))

    print('========== Check setting: Epoch: {}, Batch_size: {}, Lambda: {}, BitErrorRate: {}, LR: {}=========='.format(cfg.epochs, cfg.batch_size, lb, ber, cfg.learning_rate))
    print("========== Start training the parameter of the input transform by using one specific perturbed model ==========")

    model_np, model, model_perturbed = model_np.to(device), model.to(device), model_perturbed.to(device)

    # Hook the post-output activation for each layers.
    activation_c, activation_p = {}, {}
    for name, layer in model_np.named_modules():
        if 'relu' in name:
            layer.register_forward_hook(get_activation_c(activation_c, name))
        
    for name, layer in model_perturbed.named_modules():
        if 'relu' in name:
            layer.register_forward_hook(get_activation_p(activation_p, name))

    activationDiffDict = {}
    activationCleanDict = {}
    activationPerturbedDict = {}
    
    for epoch in range(cfg.epochs):
        running_loss = 0
        running_correct_orig = 0
        running_correct_p = 0

        '''
        # Reset the list for storing the activation:
        for name, layer in model_np.named_modules():
            if 'relu' in name:
                activationDiffDict[name] = []
                activationCleanDict[name] = []
                activationPerturbedDict[name] = []
        
        # Reset the list for the output distribution
        outputDiffList = []
        outputCleanList = []
        outputPerturbedList = []

        #if epoch % 2 == 0:
        #    checkActivationDistribution(model_np, model_perturbed, trainloader, gen, activation_c, activation_p, activationDiffDict, activationCleanDict, activationPerturbedDict, epoch)
        #    # checkOutputDistribution(model_np, model_perturbed, trainloader, gen, outputDiffList, outputCleanList, outputPerturbedList, epoch)
        '''
  
        # For each epoch, we will use N perturbed model for training.
        for batch_id, (image, label) in tqdm.tqdm(enumerate(trainloader)):
            
            total_loss = 0  

            # Model setting
            model_np.eval()
            model.eval()
            model_perturbed.eval()
            Gen.train()

            # Image setting
            image, label = image.to(device), label.to(device)
            image_adv, _ = Gen(image)
            image_adv = image_adv.to(device)

            # model inference
            _, out, out_biterror = model_np(image), model(image_adv), model_perturbed(image_adv)

            # Calculate the loss
            loss_orig, pred_orig = compute_loss(out, label)
            loss_p, pred_p = compute_loss(out_biterror, label)

            running_correct_orig += torch.sum(pred_orig == label.data).item()
            running_correct_p += torch.sum(pred_p == label.data).item()
            
            # Calculate the total loss. 
            if cfg.layerwise:
                #print(layerwise(activation_c, activation_c))
                loss = loss_orig + lb * (loss_p + layerwise(activation_c, activation_p))
            else:

                '''
                # First random sample a new batch to calculate the diversity loss.
                idx = 0
                while True:
                    idx = torch.randint(len(trainloader), (1,))
                    if idx != batch_id: break

                for random_id, (randomCleanImg, _) in enumerate(trainloader):
                    if random_id == idx: 
                        randomCleanImg = randomCleanImg.to(device)
                        diversity_loss = compute_diversity_loss(image, randomCleanImg, Gen, device)
                        break
                loss = loss_orig + lb * loss_p + 0.1 * diversity_loss
                '''
                
                loss = loss_orig + lb * loss_p 

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()        

        accuracy_orig = running_correct_orig / len(trainloader.dataset)
        accuracy_p = running_correct_p / (len(trainloader.dataset))
        print(
            "For epoch: {}, loss: {:.6f}, accuracy for clean model:"
            "{:.5f}, accuracy for perturbed model: {:.5f}".format(
                epoch + 1,
                running_loss,
                accuracy_orig,
                accuracy_p,
            )
        )

        storeLoss.append(running_loss)

        # Save the Generator model
        if (epoch + 1) % 50 == 0 or (epoch + 1) == cfg.epochs:
            # Saving the result of the generator!
            torch.save(Gen,
                cfg.save_dir + 'Single_GeneratorV9_arch_{}_LR{}_E_{}_ber_{}_lb_{}_NOWE_{}.pt'.format(arch, cfg.learning_rate, cfg.epochs, ber, lb, epoch+1))

    '''
    # For the final testing!
    # Reset the list for storing the activation:
    for name, layer in model_np.named_modules():
        if 'relu' in name:
            activationDiffDict[name] = []
            activationCleanDict[name] = []
            activationPerturbedDict[name] = []

    # checkActivationDistribution(model_np, model_perturbed, trainloader, Gen, activation_c, activation_p, activationDiffDict, activationCleanDict, activationPerturbedDict, cfg.epochs)
    

    # Reset the list for the output distribution
    outputDiffList = []
    outputCleanList = []
    outputPerturbedList = []

    # checkOutputDistribution(model_np, model_perturbed, trainloader, gen, outputDiffList, outputCleanList, outputPerturbedList, cfg.epochs)

    '''

    # Draw learning curve:
    plt.plot([e+1 for e in range(cfg.epochs)], storeLoss)
    plt.title('Learning Curve')
    plt.savefig(cfg.save_dir + 'Learning_Curve.jpg')

    print('========== Start checking the accuracy with different perturbed model ==========')

    model, _, model_perturbed, _ = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=seed)
    model, model_perturbed = model.to(device), model_perturbed.to(device)
    model.eval()
    model_perturbed.eval()
    Gen.eval()

    # Without using transform
    accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Gen, device, use_transform=False)

    # With input transform
    accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Gen, device, use_transform=True)

  