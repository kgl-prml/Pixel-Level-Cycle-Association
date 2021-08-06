import torch
import argparse
import os
import numpy as np
from torch.backends import cudnn
from config.config import cfg, cfg_from_file, cfg_from_list
import data.transforms as T
import sys
import pprint
import random
from solver.solver import Solver
from model import segmentation as SegNet
from model.domain_bn import DomainBN
from model.discriminator import FCDiscriminator
import data.datasets as Dataset
from data import utils as data_utils
from data.label_map import get_label_map
import utils.utils as gen_utils
from utils.utils import freeze_BN
from torch.nn.parallel import DistributedDataParallel
#import apex
#from apex.parallel import DistributedDataParallel


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters',
                        default=None, type=str)
    parser.add_argument('--resume', dest='resume',
                        help='initialize with saved solver status',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--local_rank', dest='local_rank',
                        help='optional local rank',
                        default=0, type=int)

    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name', 
                        default='exp', type=str)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_transform(train, dataset_name):
    base_size = cfg.DATA_TRANSFORM.LOADSIZE
    crop_size = cfg.DATA_TRANSFORM.CROPSIZE
    ignore_label = cfg.DATASET.IGNORE_LABEL

    if dataset_name == cfg.DATASET.SOURCE:
        input_size = cfg.DATA_TRANSFORM.INPUT_SIZE_S
    else:
        input_size = cfg.DATA_TRANSFORM.INPUT_SIZE_T

    min_size = int((1.0 if train else 1.0) * base_size)
    max_size = int((1.3 if train else 1.0) * base_size)

    transforms = []
    if cfg.DATA_TRANSFORM.RANDOM_RESIZE_AND_CROP:
        if train:
            transforms.append(T.RandomResize(min_size, max_size))
            transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(T.RandomCrop(crop_size, ignore_label=ignore_label))
        else:
            transforms.append(T.Resize(cfg.DATA_TRANSFORM.INPUT_SIZE_T, True))
    else:
        if train:
            transforms.append(T.Resize(input_size))
            transforms.append(T.RandomHorizontalFlip(0.5))
        else:
            transforms.append(T.Resize(input_size, True))

    mapping = get_label_map(cfg.DATASET.SOURCE, cfg.DATASET.TARGET)
    transforms.append(T.LabelRemap(mapping[dataset_name]))
    transforms.append(T.ToTensor(cfg.DATASET.IMG_MODE))
    if cfg.DATASET.IMG_MODE == "BGR":
        mean = (104.00698793, 116.66876762, 122.67891434)
        std = (1.0, 1.0, 1.0)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    transforms.append(T.Normalize(mean, std))
    return T.Compose(transforms)

def prepare_data(args):
    train_transform_S = get_transform(train=True, dataset_name=cfg.DATASET.SOURCE)
    train_transform_T = get_transform(train=True, dataset_name=cfg.DATASET.TARGET)
    val_transform = get_transform(train=False, dataset_name=cfg.DATASET.VAL)

    train_dataset_S = eval('Dataset.%s'%cfg.DATASET.SOURCE)(
            cfg.DATASET.DATAROOT_S, 
            cfg.DATASET.TRAIN_SPLIT_S, 
            transform=train_transform_S)

    train_dataset_T = eval('Dataset.%s'%cfg.DATASET.TARGET)(
            cfg.DATASET.DATAROOT_T, 
            cfg.DATASET.TRAIN_SPLIT_T, 
            transform=train_transform_T)

    val_dataset = eval('Dataset.%s'%cfg.DATASET.VAL)(
            cfg.DATASET.DATAROOT_VAL,
            cfg.DATASET.VAL_SPLIT,
            transform=val_transform)

    # construct dataloaders
    train_dataloader_S = data_utils.get_dataloader(
            train_dataset_S, cfg.TRAIN.TRAIN_BATCH_SIZE, cfg.NUM_WORKERS,
            train=True, distributed=args.distributed, 
            world_size=gen_utils.get_world_size())

    train_dataloader_T = data_utils.get_dataloader(
            train_dataset_T, cfg.TRAIN.TRAIN_BATCH_SIZE, cfg.NUM_WORKERS,
            train=True, distributed=args.distributed, 
            world_size=gen_utils.get_world_size())

    val_dataloader = data_utils.get_dataloader(
            val_dataset, cfg.TRAIN.VAL_BATCH_SIZE, cfg.NUM_WORKERS,
            train=False, distributed=args.distributed, 
            world_size=gen_utils.get_world_size())

    dataloaders = {'train_S': train_dataloader_S, \
            'train_T': train_dataloader_T, 'val': val_dataloader}

    return dataloaders

def init_net_D(args, state_dict=None):
    net_D = FCDiscriminator(cfg.DATASET.NUM_CLASSES)

    if args.distributed:
        net_D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_D)

    if cfg.MODEL.DOMAIN_BN:
        net_D = DomainBN.convert_domain_batchnorm(net_D, num_domains=2)

    if state_dict is not None:
        try:
            net_D.load_state_dict(state_dict)
        except:
            net_D = DomainBN.convert_domain_batchnorm(net_D, num_domains=2)
            net_D.load_state_dict(state_dict)

    if cfg.TRAIN.FREEZE_BN:
        net_D.apply(freeze_BN)

    if torch.cuda.is_available():
        net_D.cuda()

    if args.distributed:
        net_D = DistributedDataParallel(net_D, device_ids=[args.gpu])
    else:
        net_D = torch.nn.DataParallel(net_D)

    return net_D

def train(args):
    #seed = 12345
    #random.seed(seed)
    #np.random.seed(seed)
    #torch.random.manual_seed(seed)

    # initialize model
    model_state_dict = None
    model_state_dict_D = None
    resume_dict = None

    if cfg.RESUME != '':
        resume_dict = torch.load(cfg.RESUME, torch.device('cpu'))
        model_state_dict = resume_dict['model_state_dict']
    elif cfg.WEIGHTS != '':
        param_dict = torch.load(cfg.WEIGHTS, torch.device('cpu'))
        model_state_dict = param_dict['weights']
        model_state_dict_D = param_dict['weights_D'] if 'weights_D' in param_dict else None

    net = SegNet.__dict__[cfg.MODEL.NETWORK_NAME](
            pretrained=False, pretrained_backbone=False,
            num_classes=cfg.DATASET.NUM_CLASSES, 
            aux_loss=cfg.MODEL.USE_AUX_CLASSIFIER
            )

    net = gen_utils.load_model(net, './model/resnet101-imagenet.pth', True)

    if args.distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        #net = apex.parallel.convert_syncbn_model(net)

    if cfg.MODEL.DOMAIN_BN:
        net = DomainBN.convert_domain_batchnorm(net, num_domains=2)

    if model_state_dict is not None:
        try:
            net.load_state_dict(model_state_dict)
        except:
            net = DomainBN.convert_domain_batchnorm(net, num_domains=2)
            net.load_state_dict(model_state_dict)

    if cfg.TRAIN.FREEZE_BN:
        net.apply(freeze_BN)

    if torch.cuda.is_available():
        net.cuda()

    if args.distributed:
        net = DistributedDataParallel(net, device_ids=[args.gpu])
        #net = DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)

    net_D = init_net_D(args, model_state_dict_D) if cfg.TRAIN.ADV_TRAIN else None

    dataloaders = prepare_data(args)

    # initialize solver
    train_solver = Solver(net, net_D, dataloaders, args.distributed, 
            resume=resume_dict)

    # train 
    train_solver.solve()

    print('Finished!')

if __name__ == '__main__':
    cudnn.benchmark = True 
    args = parse_args()

    gen_utils.init_distributed_mode(args)

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.resume is not None:
        cfg.RESUME = args.resume 
    if args.weights is not None:
        cfg.WEIGHTS = args.weights
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name 

    print('Using config:')
    pprint.pprint(cfg)

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    if not os.path.exists(cfg.SAVE_DIR):
        os.makedirs(cfg.SAVE_DIR)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)

    train(args)
