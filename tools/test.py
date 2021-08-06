import torch
import argparse
from PIL import Image
import os
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from math import ceil as ceil
from torch.backends import cudnn
from config.config import cfg, cfg_from_file, cfg_from_list
import torch.nn.functional as F
import data.transforms as T
from solver.loss import AssociationLoss
import sys
import pprint
from model import segmentation as SegNet
from model.domain_bn import DomainBN
from torch.nn.parallel import DistributedDataParallel
import data.datasets as Dataset
from data import utils as data_utils
from data.label_map import get_label_map, LABEL_TASK
from utils import utils as gen_utils

colors = loadmat('data/color150.mat')['colors']

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

label_map_syn = {0: 10, 1: 2, 2: 0, 3: 1, 4: 4, 5: 8, 6: 5, 7: 13, 8: 7, 9: 11, 10: 18, 11: 17, 12: 6, 13: 12, 14: 15, 15: 3} 
label_map_gtav = {0: 10, 1: 2, 2: 0, 3: 1, 4: 4, 5: 8, 6: 5, 7: 13, 8: 7, 9: 11, 10: 18, 11: 17, 12: 6, 13: 12, 14: 15, 15: 3, 16: 9, 17: 14, 18: 16}

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters',
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

def get_transform(dataset_name):
    base_size = cfg.DATA_TRANSFORM.LOADSIZE
    ignore_label = cfg.DATASET.IGNORE_LABEL

    min_size = base_size
    max_size = base_size

    transforms = []
    transforms.append(T.Resize(cfg.DATA_TRANSFORM.INPUT_SIZE_T, True))

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
    if cfg.TEST.DOMAIN == 'source':
        dataset_name = cfg.DATASET.SOURCE
        dataset_root = cfg.DATASET.DATAROOT_S
    else:
        dataset_name = cfg.DATASET.TARGET 
        dataset_root = cfg.DATASET.DATAROOT_T

    test_transform = get_transform(dataset_name)

    dataset_split = cfg.DATASET.TEST_SPLIT
    test_dataset = eval('Dataset.%s'%dataset_name)(
            dataset_root, dataset_split, transform=test_transform)

    # construct dataloaders
    test_dataloader = data_utils.get_dataloader(
            test_dataset, cfg.TEST.BATCH_SIZE, cfg.NUM_WORKERS,
            train=False, distributed=args.distributed, 
            world_size=args.world_size)

    return test_dataset, test_dataloader

def test(args):
    # initialize model
    model_state_dict = None

    if cfg.WEIGHTS != '':
        param_dict = torch.load(cfg.WEIGHTS, 
            torch.device('cpu'))
        model_state_dict = param_dict['weights']

    net = SegNet.__dict__[cfg.MODEL.NETWORK_NAME](
            pretrained=False, pretrained_backbone=False,
            num_classes=cfg.DATASET.NUM_CLASSES, 
            aux_loss=cfg.MODEL.USE_AUX_CLASSIFIER
            )

    if args.distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if cfg.MODEL.DOMAIN_BN:
        net = DomainBN.convert_domain_batchnorm(net, num_domains=2)

    if model_state_dict is not None:
        try:
            net.load_state_dict(model_state_dict)
        except:
            net = DomainBN.convert_domain_batchnorm(net, num_domains=2)
            net.load_state_dict(model_state_dict)
            if cfg.TEST.DOMAIN == 'source':
                DomainBN.set_domain_id(net, 0)
            if cfg.TEST.DOMAIN == 'target':
                DomainBN.set_domain_id(net, 1)

    if torch.cuda.is_available():
        net.cuda()

    if args.distributed:
        net = DistributedDataParallel(net, device_ids=[args.gpu])
    else:
        net = torch.nn.DataParallel(net)

    test_dataset, test_dataloader = prepare_data(args)

    net.eval()
    corrects = 0
    total_num_pixels = 0
    total_intersection = 0
    total_union = 0
    num_classes = cfg.DATASET.NUM_CLASSES

    with torch.no_grad():
        conmat = gen_utils.ConfusionMatrix(cfg.DATASET.NUM_CLASSES, 
                    list(LABEL_TASK['%s2%s' % (cfg.DATASET.SOURCE, cfg.DATASET.TARGET)].keys()))
        for sample in iter(test_dataloader):
            data, gt = gen_utils.to_cuda(sample['Img']), gen_utils.to_cuda(sample['Label'])
            names = sample['Name']
            res = net(data) 

            if cfg.TEST.WITH_AGGREGATION:
                feats = res['feat']
                alpha = 0.5
                feats = (1.0 - alpha) * feats + alpha * AssociationLoss().spatial_agg(feats)[-1]
                preds = F.softmax(net.module.classifier(feats), dim=1)
                preds = (1.0 - alpha) * preds + alpha * AssociationLoss().spatial_agg(preds, metric='kl')[-1]
            else:
                preds = res['out']

            preds = F.interpolate(preds, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.max(preds, dim=1).indices

            if cfg.TEST.VISUALIZE:
                for i in range(preds.size(0)):
                    cur_pred = preds[i, :, :].cpu().numpy()
                    cur_gt = gt[i, :, :].cpu().numpy()
                    cur_pred_cp = cur_pred.copy()
                    cur_gt_cp = cur_gt.copy()
                    label_map = label_map_gtav if cfg.DATASET.SOURCE == 'GTAV' else label_map_syn
                    for n in range(cfg.DATASET.NUM_CLASSES):
                        cur_pred[cur_pred_cp == n] = label_map[n]
                        cur_gt[cur_gt_cp == n] = label_map[n]

                    cur_pred = np.where(cur_gt == 255, cur_gt, cur_pred)

                    cur_pred = np.asarray(cur_pred, dtype=np.uint8)
                    cur_gt = np.asarray(cur_gt, dtype=np.uint8)

                    vis_res = colorize_mask(cur_pred)
                    vis_gt = colorize_mask(cur_gt)

                    vis_name = 'vis_%s.png'%(names[i])
                    vis_res.save(os.path.join(cfg.SAVE_DIR, vis_name))

                    vis_name = 'vis_gt_%s.png'%(names[i])
                    vis_gt.save(os.path.join(cfg.SAVE_DIR, vis_name))

            conmat.update(gt.flatten(), preds.flatten())

        conmat.reduce_from_all_processes()
        print('Test with %d samples: ' % len(test_dataset))
        print(conmat)

    print('Finished!')

if __name__ == '__main__':
    cudnn.benchmark = True 
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

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

    args.world_size = 1
    gen_utils.init_distributed_mode(args)
    test(args)
