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
