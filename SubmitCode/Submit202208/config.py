from easydict import EasyDict

cfg = EasyDict()

cfg.faulty_layers = ['conv', 'l1', 'l2', 'l3', 'l4', 'linear']  # For ResNet18, other only [conv, linear].

cfg.batch_size = 512
cfg.test_batch_size = 100
cfg.epochs = 2
cfg.precision = 8

# For setting the machine
cfg.data_dir = (
    "dataset"
)
cfg.model_dir = (
    "model_weights/symmetric_signed/"
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
cfg.N = 10
cfg.randomRange = 30000
cfg.totalRandom = True # True: Sample perturbed models in the range cfg.randomRange
cfg.G = 'ConvL'

# For transform generalization testing:
cfg.beginSeed = 50000
cfg.endSeed = 50010

# For transform_eval
cfg.testing_mode = 'visualization' # clean / generator_base / visualization
cfg.G_PATH = '.'

# For visualization
cfg.tsneModel = 50000

