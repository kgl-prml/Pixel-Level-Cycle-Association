"""
 Created by Guoliang Kang. 
 This multi-gpu version is modified from https://github.com/bermanmaxim/LovaszSoftmax.
"""

import torch
from utils.utils import to_cuda, to_onehot, get_rank
import torch.distributed as dist
from torch.autograd import Variable
import torch.nn.functional as F
from utils.utils import to_cuda, to_onehot, get_rank
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=255):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels), classes=classes, ignore=ignore)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present', ignore=255):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes

    num_gpus = dist.get_world_size()
    rank = get_rank()
    labels_collect = []
    probas_collect = []
    for r in range(num_gpus):
        labels_collect.append(to_cuda(torch.ones(labels.size()).long()))
        probas_collect.append(to_cuda(torch.ones(probas.size())))

    labels_collect[rank] = labels.clone()
    probas_collect[rank] = probas.clone()

    for r in range(num_gpus):
        dist.broadcast(labels_collect[r], src=r)
        dist.broadcast(probas_collect[r], src=r)

    num_valids = []
    for r in range(num_gpus):
        num_valids.append(torch.sum(labels_collect[r] != 255).item())
    num_valids = np.cumsum(num_valids)

    labels_collect = torch.cat(labels_collect, dim=0).detach()
    probas_collect = torch.cat(probas_collect, dim=0).detach()

    valid_labels = (labels_collect != 255)
    assert(torch.sum(valid_labels).item() == num_valids[-1])
    labels_collect = labels_collect[valid_labels]
    probas_collect = probas_collect[valid_labels.nonzero().squeeze()]

    lg_collect_cls = {}
    start = 0 if rank == 0 else num_valids[rank-1]
    end = num_valids[rank]

    for c in class_to_sum:
        fg_collect = (labels_collect == c).float()
        if (classes == 'present' and fg_collect.sum() == 0):
            continue

        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred_collect = probas_collect[:, 0]
        else:
            class_pred_collect = probas_collect[:, c]

        errors_collect = (fg_collect - class_pred_collect).abs()

        _, perm = torch.sort(errors_collect, 0, descending=True)
        perm = perm.data
        fg_collect_sorted = fg_collect[perm]
        lg_collect = lovasz_grad(fg_collect_sorted)
        assert(num_valids[-1] == lg_collect.size(0))

        lg_collect = to_cuda(torch.zeros(lg_collect.size())).scatter_(0, perm,
                              lg_collect).detach()

        #errors_collect = to_cuda(torch.zeros(errors_collect.size())).scatter_(0, perm,
        #                      errors_collect).detach()
        #
        #lg = lg_collect[start:end].data
        #errors = errors_collect[start:end]
        #losses.append(torch.dot(errors, lg))

        lg_collect_cls[c] = lg_collect

    #print(num_valids)
    valid = (labels != 255)
    labels = labels[valid]
    probas = probas[valid.nonzero().squeeze()]

    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        #return probas * 0.
        return 0.0

    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c

        if (classes == 'present' and fg.sum() == 0):
            continue

        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]

        errors = (fg - class_pred).abs()

        lg = lg_collect_cls[c][start:end].data
        losses.append(torch.dot(errors, lg))

    return mean(losses) * num_gpus


def flatten_probas(probas, labels):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    return probas, labels
    #valid = (labels != ignore)
    #vprobas = probas[valid.nonzero().squeeze()]
    #vlabels = labels[valid]
    #return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

