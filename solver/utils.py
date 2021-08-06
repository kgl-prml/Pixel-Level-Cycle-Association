import torch
import torch.nn.functional as F
import torch.nn as nn
import utils.utils as gen_utils
import numpy as np

def adjust_rate_poly(cur_iter, max_iter, power=0.9):
    return (1.0 - 1.0 * cur_iter / max_iter) ** power

def adjust_learning_rate_exp(lr, optimizer, iters, decay_rate=0.1, decay_step=25):
    lr = lr * (decay_rate ** (iters // decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_RevGrad(lr, optimizer, max_iter, cur_iter, 
        alpha=10, beta=0.75):
    p = 1.0 * cur_iter / (max_iter - 1)
    lr = lr / pow(1.0 + alpha * p, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_inv(lr, optimizer, iters, alpha=0.001, beta=0.75):
    lr = lr / pow(1.0 + alpha * iters, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_step(lr, optimizer, iters, steps, beta=0.1):
    n = 0
    for step in steps:
        if iters < step:
            break
        n += 1

    lr = lr * (beta ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_poly(lr, optimizer, iters, max_iter, power=0.9):
    lr = lr * (1.0 - 1.0 * iters / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def set_param_groups(net, lr_mult_dict={}):
    params = []
    if hasattr(net, "module"):
        net = net.module

    modules = net._modules
    for name in modules:
        module = modules[name]
        if name in lr_mult_dict:
            params += [{'params': module.parameters(), \
                    'lr_mult': lr_mult_dict[name]}]
        else:
            params += [{'params': module.parameters(), 'lr_mult': 1.0}]

    return params

def LSR(x, dim=1, thres=10.0):
    lsr = -1.0 * torch.mean(x, dim=dim)
    if thres > 0.0:
        return torch.mean((lsr/thres-1.0).detach() * lsr)
    else:
        return torch.mean(lsr)

def crop(feats, preds, gt, h, w):
    H, W = feats.shape[-2:]
    tmp_feats = []
    tmp_preds = []
    tmp_gt = []
    N = feats.size(0)
    for i in range(N):
        inds_H = torch.randperm(H)[0:h]
        inds_W = torch.randperm(W)[0:w]
        tmp_feats += [feats[i, :, inds_H[:, None], inds_W]]
        tmp_preds += [preds[i, :, inds_H[:, None], inds_W]]
        tmp_gt += [gt[i, inds_H[:, None], inds_W]]

    new_feats = torch.stack(tmp_feats, dim=0)
    new_gt = torch.stack(tmp_gt, dim=0)
    new_preds = torch.stack(tmp_preds, dim=0)
    probs = F.softmax(new_preds, dim=1)
    return new_feats, probs, new_gt

