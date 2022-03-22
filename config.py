import os
from easydict import EasyDict

cfg = EasyDict()

cfg.faulty_layers = []
# cfg.faulty_layers = ['linear']
# cfg.faulty_layers = ['linear', 'conv']

cfg.batch_size = 128
cfg.test_batch_size = 100
cfg.epochs = 5
cfg.precision = 8
# cfg.net = 'resnet56'
# cfg.dataset = 'cifar10'
cfg.models_dir = "./models"
cfg.data_dir = "~/barn-shared/datasets"
cfg.work_dir = "~/scratch/energy-efficient-resilience"
cfg.save_dir = "~/barn/energy-efficient-resilience-save-dir"
cfg.temperature = 1

cfg.channels = 3
cfg.w1 = 32  # 28 #224
cfg.h1 = 32  # 28 #224
cfg.w2 = 32  # 32 28 #224
cfg.h2 = 32  # 32 28 #224
cfg.lmd = 5e-7
cfg.learning_rate = 1e-3
cfg.flow_lr = 2e-3
cfg.decay = 0.96
cfg.max_epoch = 1
cfg.lb = 1

if not os.path.exists(cfg.save_dir):
    os.makedirs(cfg.save_dir)
