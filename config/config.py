import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset options
#
__C.DATASET = edict()
__C.DATASET.SOURCE = 'SYNTHIA'
__C.DATASET.TARGET = 'Cityscapes'
__C.DATASET.VAL = 'Cityscapes'
__C.DATASET.DATAROOT_S = ''
__C.DATASET.DATAROOT_T = ''
__C.DATASET.DATAROOT_VAL = ''

__C.DATASET.NUM_CLASSES = 0
__C.DATASET.TRAIN_SPLIT_S = ''
__C.DATASET.TRAIN_SPLIT_T = ''
__C.DATASET.VAL_SPLIT = ''
__C.DATASET.TEST_SPLIT = ''
__C.DATASET.IMG_MODE = 'BGR'

__C.DATASET.IGNORE_LABEL = 255

# Model options
#
__C.MODEL = edict()
__C.MODEL.NETWORK_NAME = 'deeplabv3_resnet101'
__C.MODEL.USE_AUX_CLASSIFIER = False
__C.MODEL.DOMAIN_BN = False
__C.MODEL.FEAT_DIM = 2048

# data pre-processing options
#
__C.DATA_TRANSFORM = edict()
__C.DATA_TRANSFORM.LOADSIZE = 1024
__C.DATA_TRANSFORM.CROPSIZE = 796
__C.DATA_TRANSFORM.INPUT_SIZE_S = (720, 1280)
__C.DATA_TRANSFORM.INPUT_SIZE_T = (760, 1520)

__C.DATA_TRANSFORM.RANDOM_RESIZE_AND_CROP = True

# Training options
#
__C.TRAIN = edict()
# batch size setting
__C.TRAIN.METHOD = ''
__C.TRAIN.USE_CROP = False
__C.TRAIN.USE_DOWNSAMPLING = False
__C.TRAIN.SCALE_FACTOR = 0.2
__C.TRAIN.LOV_W = 0.75
__C.TRAIN.ASSO_W = 0.1
__C.TRAIN.LSR_W = 0.01
__C.TRAIN.APPLY_SPAGG = True
__C.TRAIN.SPAGG_ALPHA = 0.5
__C.TRAIN.APPLY_MULTILAYER_ASSOCIATION = True
__C.TRAIN.ASSO_PRINT_INFO = False

__C.TRAIN.ASSO_TOPK = 1
__C.TRAIN.LSR_THRES = 10.0
__C.TRAIN.WITH_LOV = True

__C.TRAIN.FREEZE_BN = False

__C.TRAIN.TRAIN_BATCH_SIZE = 30
__C.TRAIN.VAL_BATCH_SIZE = 30 
__C.TRAIN.LOSS_TYPE = 'SegCrossEntropyLoss'
__C.TRAIN.DS_WEIGHTS = (1.0, 0.4)

# learning rate schedule
__C.TRAIN.BASE_LR = 0.001
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.BASE_LR_D = 0.0001
__C.TRAIN.MOMENTUM_D = 0.9
__C.TRAIN.LR_MULT = 10.0
__C.TRAIN.OPTIMIZER = 'SGD'
__C.TRAIN.OPTIMIZER_D = 'Adam'
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.WEIGHT_DECAY_D = 0.0005
__C.TRAIN.LR_SCHEDULE = 'poly'
__C.TRAIN.MAX_EPOCHS = 50
__C.TRAIN.LOGGING = True
# percentage of total iterations each epoch
__C.TRAIN.TEST_INTERVAL = 1.0  
# percentage of total iterations in each epoch
__C.TRAIN.SAVE_CKPT_INTERVAL = 10.0  
__C.TRAIN.NUM_LOGGING_PER_EPOCH = 10.0
__C.TRAIN.ITER_SIZE = 1
__C.TRAIN.ADV_W = 0.001
__C.TRAIN.ADV_TRAIN = False

# optimizer options
__C.ADAM = edict()
__C.ADAM.BETA1 = 0.9
__C.ADAM.BETA2 = 0.999
__C.ADAM.BETA1_D = 0.9
__C.ADAM.BETA2_D = 0.999

__C.INV = edict()
__C.INV.ALPHA = 0.001
__C.INV.BETA = 0.75

__C.EXP = edict()
__C.EXP.LR_DECAY_RATE = 0.1
__C.EXP.LR_DECAY_STEP = 30

__C.POLY = edict()
__C.POLY.POWER = 0.9
__C.POLY.MAX_EPOCHS = 70

__C.STEP = edict()
__C.STEP.STEPS = ()
__C.STEP.BETA = 0.1

# Testing options
#
__C.TEST = edict()
__C.TEST.BATCH_SIZE = 30
__C.TEST.DOMAIN = ""
__C.TEST.VISUALIZE = False
__C.TEST.WITH_AGGREGATION = True

# MISC
__C.WEIGHTS = ''
__C.RESUME = ''
__C.EVAL_METRIC = "mIoU" # "accuracy" as alternative
__C.EXP_NAME = 'exp'
__C.SAVE_DIR = ''
__C.NUM_WORKERS = 3

__C.ENGINE = edict()
__C.ENGINE.LOCAL_RANK = 0

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k in a:
        # a must specify keys that are in b
        v = a[k]
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
