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

from easydict import EasyDict

cfg = EasyDict()

cfg.machine = 'lab' # TWCC / lab

# cfg.faulty_layers = []
# cfg.faulty_layers = ['l3']
cfg.faulty_layers = ['conv', 'l1', 'l2', 'l3', 'l4', 'linear']  # For ResNet18
# cfg.faulty_layers = ["linear", "conv"]

cfg.batch_size = 512
cfg.test_batch_size = 100
cfg.epochs = 2
cfg.precision = 8

# For setting the machine
if cfg.machine == 'TWCC':
    cfg.data_dir = (
        "/home/u7590150/dataset"
    )
    cfg.model_dir = (
        "model_weights/symmetric_signed/"
    )
    cfg.save_dir = (
        "/home/u7590150/tmp/"
    )
    cfg.save_dir_curve = (
        "/home/u7590150/tmp_curve/"
    )
else:
    cfg.data_dir = (
        "/home/haolun/dataset"
    )
    cfg.model_dir = (
        "model_weights/symmetric_signed/"
    )
    cfg.save_dir = (
        "/home/haolun/tmp/"
    )
    cfg.save_dir_curve = (
        "/home/haolun/tmp_curve/"
    )
    

cfg.temperature = 1
cfg.channels = 3

cfg.w1 = 32  # 28 #224
cfg.h1 = 32  # 28 #224
cfg.w2 = 32  # 32 28 #224
cfg.h2 = 32  # 32 28 #224
cfg.lmd = 5e-7
cfg.learning_rate = 1
cfg.flow_lr = 2e-3
cfg.decay = 0.96
cfg.max_epoch = 1
cfg.lb = 1
cfg.device = None
cfg.seed = 0


# For EOPM
cfg.N = 100
cfg.randomRange = 30000
cfg.totalRandom = True # True: Sample perturbed models in the range cfg.randomRange
cfg.layerwise = False # True: Layerwise training / False: Normal training
cfg.G = 'large'

# For transform adversarial
cfg.alpha = 1
cfg.PGD_STEP = 1

# For transform generalization testing:
cfg.beginSeed = 50000
cfg.endSeed = 50010

# For transform_eval
cfg.testing_mode = 'clean' # clean / random_bit_error / adversarial / activation / generator_base
cfg.P_PATH = '/home/haolun/energy-efficient-resilience_dev/activation_p/Activation_test_arch_resnet18_LR_p0.00075_E_5_ber_0.01_lb_0.01_X+P_NowE5.pt'
# cfg.G_PATH = '/home/haolun/energy-efficient-resilience_dev_forRandomImprove/generatorBackup/EOPM_GeneratorV1_arch_resnet18_LR0.001_E_300_ber_0.01_lb_5.0_N_10_step1000_NOWE_300.pt'
cfg.G_PATH = '/home/haolun/energy-efficient-resilience_dev_forRandomImprove/adversarial_gen_bit/Adversarial_GeneratorV1_cifar10_arch_resnet18_LR_a1_p0.01_E_300_PGD_1_ber_0.01_lb_1.0_NOWE_300_bit_1000.pt'

