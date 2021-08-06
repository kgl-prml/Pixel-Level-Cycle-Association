import torch
import torch.nn as nn
import os 
from config.config import cfg
from math import ceil as ceil
from torch import optim
from . import utils as solver_utils
import utils.utils as gen_utils
from utils.utils import to_cuda, get_world_size
from .loss import SegCrossEntropyLoss, AssociationLoss 
from .lov_softmax_multigpu import lovasz_softmax as lovasz_softmax_multigpu
from .lov_softmax import lovasz_softmax
from model.domain_bn import DomainBN
import torch.nn.functional as F
import numpy as np

class Solver(object):
    def __init__(self, net, net_D, dataloaders, distributed=False, resume=None, **kwargs):
        self.net = net
        self.net_D = net_D
        self.adv_train = (self.net_D is not None and cfg.TRAIN.ADV_TRAIN)

        self.distributed = distributed
        self.iter_size = cfg.TRAIN.ITER_SIZE

        self.init_data(dataloaders)

        self.CELoss = eval(cfg.TRAIN.LOSS_TYPE)(
                ignore_index=cfg.DATASET.IGNORE_LABEL,
                ds_weights=cfg.TRAIN.DS_WEIGHTS)

        self.BCELoss = torch.nn.BCEWithLogitsLoss()

        self.FeatAssociationLoss = AssociationLoss(metric='cos', 
                spagg=cfg.TRAIN.APPLY_SPAGG, spagg_alpha=cfg.TRAIN.SPAGG_ALPHA, 
                asso_topk=cfg.TRAIN.ASSO_TOPK, print_info=cfg.TRAIN.ASSO_PRINT_INFO)

        self.ClsAssociationLoss = AssociationLoss(metric='kl', 
                spagg=cfg.TRAIN.APPLY_SPAGG, spagg_alpha=cfg.TRAIN.SPAGG_ALPHA, 
                asso_topk=cfg.TRAIN.ASSO_TOPK, print_info=cfg.TRAIN.ASSO_PRINT_INFO)

        if torch.cuda.is_available():
            self.CELoss.cuda()
            self.BCELoss.cuda()

        self.optim_state_dict = None
        self.optim_state_dict_D = None
        self.resume = False
        self.epochs = 0
        self.iters = 0
        if resume is not None:
            self.resume = True
            self.epochs = resume['epochs']
            self.iters = resume['iters']
            self.optim_state_dict = resume['optimizer_state_dict']
            if 'optimizer_state_dict_D' in resume:
                self.optim_state_dict_D = resume['optimizer_state_dict_D']
            print('Resume Training from epoch %d, iter %d.' % \
			(self.epochs, self.iters))

        self.base_lr = cfg.TRAIN.BASE_LR
        self.momentum = cfg.TRAIN.MOMENTUM
        self.build_optimizer()

        if self.adv_train:
            self.base_lr_D = cfg.TRAIN.BASE_LR_D
            self.momentum_D = cfg.TRAIN.MOMENTUM_D
            self.build_optimizer_D()

    def init_data(self, dataloaders):
        self.train_data = dict()
        self.train_data['loader_S'] = dataloaders['train_S']
        self.train_data['loader_T'] = dataloaders['train_T']
        self.train_data['iterator_S'] = None
        self.train_data['iterator_T'] = None

        if 'val' in dataloaders:
            self.test_data = dict()
            self.test_data['loader'] = dataloaders['val']

    def build_optimizer(self):
        param_groups = solver_utils.set_param_groups(self.net, 
            {'classifier': cfg.TRAIN.LR_MULT, 
             'aux_classifier': cfg.TRAIN.LR_MULT})

        assert cfg.TRAIN.OPTIMIZER in ["Adam", "SGD"], \
            "Currently do not support your specified optimizer."

        if cfg.TRAIN.OPTIMIZER == "Adam":
            self.optimizer = optim.Adam(param_groups, 
			lr=self.base_lr, betas=[cfg.ADAM.BETA1, cfg.ADAM.BETA2], 
			weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        elif cfg.TRAIN.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(param_groups, 
			lr=self.base_lr, momentum=self.momentum, 
			weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        if self.optim_state_dict is not None:
            self.optimizer.load_state_dict(self.optim_state_dict)

    def build_optimizer_D(self):
        param_groups = solver_utils.set_param_groups(self.net_D) 

        assert cfg.TRAIN.OPTIMIZER_D in ["Adam", "SGD"], \
            "Currently do not support your specified optimizer."

        if cfg.TRAIN.OPTIMIZER_D == "Adam":
            self.optimizer_D = optim.Adam(param_groups, 
			lr=self.base_lr_D, betas=[cfg.ADAM.BETA1_D, cfg.ADAM.BETA2_D], 
			weight_decay=cfg.TRAIN.WEIGHT_DECAY_D)

        elif cfg.TRAIN.OPTIMIZER_D == "SGD":
            self.optimizer_D = optim.SGD(param_groups, 
			lr=self.base_lr_D, momentum=self.momentum_D, 
			weight_decay=cfg.TRAIN.WEIGHT_DECAY_D)

        if self.optim_state_dict_D is not None:
            self.optimizer_D.load_state_dict(self.optim_state_dict_D)

    def update_lr(self, optimizer=None, base_lr=None):
        iters = self.iters
        max_iters = self.max_iters
        if optimizer is None:
            optimizer = self.optimizer

        if base_lr is None:
            base_lr = self.base_lr
        
        if cfg.TRAIN.LR_SCHEDULE == 'exp':
            solver_utils.adjust_learning_rate_exp(base_lr, 
			optimizer, iters, 
                        decay_rate=cfg.EXP.LR_DECAY_RATE,
			decay_step=cfg.EXP.LR_DECAY_STEP)

        elif cfg.TRAIN.LR_SCHEDULE == 'inv':
            solver_utils.adjust_learning_rate_inv(base_lr, optimizer, 
		    iters, cfg.INV.ALPHA, cfg.INV.BETA)

        elif cfg.TRAIN.LR_SCHEDULE == 'step':
            steps = cfg.STEP.STEPS
            beta = cfg.STEP.BETA
            solver_utils.adjust_learning_rate_step(base_lr, optimizer, 
                    iters, steps, beta)

        elif cfg.TRAIN.LR_SCHEDULE == 'poly':
            max_iters = cfg.POLY.MAX_EPOCHS * self.iters_per_epoch
            solver_utils.adjust_learning_rate_poly(base_lr, optimizer, iters, max_iters, power=cfg.POLY.POWER)

        elif cfg.TRAIN.LR_SCHEDULE == 'fixed':
            pass

        else:
            raise NotImplementedError("Currently don't support the specified \
                    learning rate schedule: %s." % cfg.TRAIN.LR_SCHEDULE)

    def logging(self, loss, res):
        print('[epoch: %d, iter: %d]: ' % (self.epochs, self.iters))
        info_str = gen_utils.format_dict(loss) + '; ' + gen_utils.format_dict(res)
        print(info_str)

    def save_ckpt(self, complete=False):
        save_path = cfg.SAVE_DIR
        if not complete:
            ckpt_resume = os.path.join(save_path, 'ckpt_%d_%d.resume' % (self.epochs, self.iters))
            ckpt_weights = os.path.join(save_path, 'ckpt_%d_%d.weights' % (self.epochs, self.iters))
        else:
            ckpt_resume = os.path.join(save_path, 'ckpt_final.resume')
            ckpt_weights = os.path.join(save_path, 'ckpt_final.weights')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if hasattr(self.net, "module"):
            net = self.net.module
        else:
            net = self.net

        to_resume = {'epochs': self.epochs,
                    'iters': self.iters,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }

        if self.adv_train:
            if hasattr(self.net_D, "module"):
                net_D = self.net_D.module
            else:
                net_D = self.net_D

            to_resume['model_state_dict_D'] = net_D.state_dict()
            to_resume['optimizer_state_dict_D'] = self.optimizer_D.state_dict()
            
        torch.save(to_resume, ckpt_resume)
        torch.save({'weights': net.state_dict()}, ckpt_weights)

    def complete_training(self):
        if self.epochs > cfg.TRAIN.MAX_EPOCHS:
            return True

    def set_domain_id(self, domain_id):
        if hasattr(self.net, "module"):
            net = self.net.module
        else:
            net = self.net

        DomainBN.set_domain_id(net, domain_id)

    def test(self):
        self.set_domain_id(1)
        self.net.eval()

        num_classes = cfg.DATASET.NUM_CLASSES
        conmat = gen_utils.ConfusionMatrix(num_classes)

        for sample in iter(self.test_data['loader']):
            data, gt = gen_utils.to_cuda(sample['Img']), gen_utils.to_cuda(sample['Label'])
            logits = self.net(data)['out']
            logits = F.interpolate(logits, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.max(logits, dim=1).indices

            conmat.update(gt.flatten(), preds.flatten())

        conmat.reduce_from_all_processes()
        accu, _, iou = conmat.compute()
        return accu.item() * 100.0, iou.mean().item() * 100.0

    def solve(self):
        if self.resume:
            self.iters += 1
            self.epochs += 1

        self.compute_iters_per_epoch()
        while True:
            if self.epochs >= cfg.TRAIN.MAX_EPOCHS: 
                break

            self.update_network()
            self.epochs += 1

        self.epochs -= 1
        self.iters -= 1
        if not self.distributed or gen_utils.is_main_process(): 
            self.save_ckpt(complete=True)
        print('Training Done!')

    def compute_iters_per_epoch(self):
        self.iters_per_epoch = ceil(1.0 * len(self.train_data['loader_T']) 
                / self.iter_size)
        self.max_iters = self.iters_per_epoch * cfg.TRAIN.MAX_EPOCHS
        print('Iterations in one epoch: %d' % (self.iters_per_epoch))

    def get_training_samples(self, domain_key):
        assert('loader_%s'%domain_key in self.train_data and \
                'iterator_%s'%domain_key in self.train_data)

        loader_key = 'loader_' + domain_key
        iterator_key = 'iterator_' + domain_key
        data_loader = self.train_data[loader_key]
        data_iterator = self.train_data[iterator_key]
        assert data_loader is not None and data_iterator is not None, \
            'Check your training dataloader.' 

        try:
            sample = next(data_iterator)
        except StopIteration:
            self.iter(domain_key)
            sample = next(self.train_data[iterator_key])

        return sample

    def iter(self, domain_key):
        if self.distributed:
           r = self.epochs #np.random.randint(0, cfg.TRAIN.MAX_EPOCHS)
           self.train_data['loader_'+domain_key].sampler.set_epoch(r)
        self.train_data['iterator_'+domain_key] = iter(self.train_data['loader_'+domain_key])
       
    def update_network(self):
        # initial configuration
        stop = False
        update_iters = 0

        self.iter('S')
        self.iter('T')

        while not stop:
            # update learning rate
            self.update_lr(self.optimizer, self.base_lr)

            # set the status of network
            self.net.train()
            self.net.zero_grad()

            if self.adv_train:
                self.update_lr(self.optimizer_D, self.base_lr_D)
                self.net_D.train()
                self.net_D.zero_grad()

            loss = 0

            for k in range(self.iter_size):
                sample_S = self.get_training_samples('S')
                data_S, gt_S = sample_S['Img'], sample_S['Label']
                data_S, gt_S = gen_utils.to_cuda(data_S), gen_utils.to_cuda(gt_S)

                sample_T = self.get_training_samples('T') 
                data_T, gt_T = sample_T['Img'], sample_T['Label']
                data_T, gt_T = gen_utils.to_cuda(data_T), gen_utils.to_cuda(gt_T)

                loss_dict, out_dict = eval('self.%s'%cfg.TRAIN.METHOD)(data_S, gt_S, data_T, gt_T)
                loss = loss_dict['total'] / self.iter_size

                preds_S, preds_T = out_dict['preds_S'], out_dict['preds_T']
                
                if self.adv_train:
                    # G step: 
                    probs_S, probs_T = F.softmax(preds_S, dim=1), F.softmax(preds_T, dim=1)
                    for param in self.net_D.parameters():
                        param.requires_grad = False

                    loss_GD = self.G_step(probs_S, probs_T) / self.iter_size
                    loss += cfg.TRAIN.ADV_W * loss_GD
                    loss_dict['G_loss'] = loss_GD

                loss.backward()

                if self.adv_train:
                    # D step:
                    for param in self.net_D.parameters():
                        param.requires_grad = True

                    loss_D = self.D_step(probs_S, probs_T) / self.iter_size
                    loss_dict['D_loss'] = loss_D
                    loss_D.backward()

            # update the network
            self.optimizer.step()
            if self.adv_train:
                # update the discriminator
                self.optimizer_D.step()

            if cfg.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_epoch // cfg.TRAIN.NUM_LOGGING_PER_EPOCH)) == 0:

                preds = out_dict['preds_S']
                accu = 100.0 * gen_utils.model_eval(torch.max(preds, dim=1).indices, gt_S, 
                           'accuracy', preds.size(1), cfg.DATASET.IGNORE_LABEL).item()
                miou = 100.0 * gen_utils.model_eval(torch.max(preds, dim=1).indices, gt_S, 
                           'mIoU', preds.size(1), cfg.DATASET.IGNORE_LABEL)[0].item()

                cur_loss = loss_dict
                eval_res = {'accu': accu, 'miou': miou}
                self.logging(cur_loss, eval_res)

            if cfg.TRAIN.TEST_INTERVAL > 0 and \
	        (self.iters+1) % int(cfg.TRAIN.TEST_INTERVAL * self.iters_per_epoch) == 0:
                with torch.no_grad():
                    accu, miou = self.test()
                print('Test at (epoch %d, iter %d) with %s.' % (
                              self.epochs, self.iters, 
                              gen_utils.format_dict({'accu': accu, 'miou': miou})
                              )
                     )

            if not self.distributed or gen_utils.is_main_process(): 
                if cfg.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
	            (self.iters+1) % int(cfg.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_epoch) == 0:
                    self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_epoch:
                stop = True
            else:
                stop = False

    def source_only(self, data_S, gt_S, data_T, gt_T, *others, **kwargs):
        self.set_domain_id(0)
        preds = self.net(data_S)['out']
        preds = F.interpolate(preds, size=data_S.shape[-2:], mode='bilinear', align_corners=False) 
        ce_loss = self.CELoss([preds], gt_S)
        if cfg.TRAIN.WITH_LOV:
            if self.distributed:
                lov_loss = lovasz_softmax_multigpu(F.softmax(preds, dim=1), gt_S, classes='present', per_image=False, ignore=255)
            else:
                lov_loss = lovasz_softmax(F.softmax(preds, dim=1), gt_S, classes='present', per_image=False, ignore=255)

            ce_loss += (cfg.TRAIN.LOV_W * get_world_size() * self.iter_size) * lov_loss

        out_dict = {'feats_S': None, 'feats_T': None, 'preds_S': preds, 'preds_T': None}
        return {'total': ce_loss}, out_dict

    def association(self, data_S, gt_S, data_T, gt_T, **kwargs):
        if cfg.MODEL.DOMAIN_BN:
            self.set_domain_id(1)
        res_T = self.net(data_T)
        preds_T = res_T['out']
        feats_T = res_T['feat']

        if cfg.MODEL.DOMAIN_BN:
            self.set_domain_id(0)
        res_S = self.net(data_S)
        preds_S = res_S['out']
        feats_S = res_S['feat']

        total_loss = 0.0
        total_loss_dict = {}

        H, W = feats_S.shape[-2:]
        new_gt_S = F.interpolate(gt_S.type(torch.cuda.FloatTensor).unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
        new_gt_T = F.interpolate(gt_T.type(torch.cuda.FloatTensor).unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if cfg.TRAIN.USE_CROP:
            scale_factor = cfg.TRAIN.SCALE_FACTOR
            N = feats_S.size(0)
            new_H, new_W = int(scale_factor * H), int(scale_factor * W)

            feats_S, probs_S, new_gt_S = solver_utils.crop(feats_S, preds_S, new_gt_S, new_H, new_W)
            feats_T, probs_T, new_gt_T = solver_utils.crop(feats_T, preds_T, new_gt_T, new_H, new_W)

        elif cfg.TRAIN.USE_DOWNSAMPLING:
            scale_factor = cfg.TRAIN.SCALE_FACTOR
            feats_S = F.interpolate(feats_S, scale_factor=scale_factor, mode='bilinear', 
                    recompute_scale_factor=False, align_corners=False)
            feats_T = F.interpolate(feats_T, scale_factor=scale_factor, mode='bilinear',
                    recompute_scale_factor=False, align_corners=False)
            new_preds_S = F.interpolate(preds_S, scale_factor=scale_factor, mode='bilinear', 
                    recompute_scale_factor=False, align_corners=False)
            new_preds_T = F.interpolate(preds_T, scale_factor=scale_factor, mode='bilinear', 
                    recompute_scale_factor=False, align_corners=False)

            H, W = feats_S.shape[-2:]
            new_gt_S = F.interpolate(gt_S.type(torch.cuda.FloatTensor).unsqueeze(1), size=(H, W), 
                    mode='nearest').squeeze(1)
            new_gt_T = F.interpolate(gt_T.type(torch.cuda.FloatTensor).unsqueeze(1), size=(H, W), 
                    mode='nearest').squeeze(1)

            probs_S, probs_T = F.softmax(new_preds_S, dim=1), F.softmax(new_preds_T, dim=1)

        else:
            probs_S, probs_T = F.softmax(preds_S, dim=1), F.softmax(preds_T, dim=1)

        ass_loss_dict = self.FeatAssociationLoss(feats_S, feats_T, new_gt_S, new_gt_T)
        ass_loss = ass_loss_dict['association']
        total_loss += cfg.TRAIN.ASSO_W * ass_loss
        total_loss_dict.update(ass_loss_dict)

        if cfg.TRAIN.APPLY_MULTILAYER_ASSOCIATION:
            ass_loss_classifier_dict = self.ClsAssociationLoss(probs_S, probs_T, new_gt_S, new_gt_T)

            ass_loss_classifier = ass_loss_classifier_dict['association']
            total_loss += cfg.TRAIN.ASSO_W * ass_loss_classifier
            ass_loss_classifier_dict = {key+'_cls': ass_loss_classifier_dict[key] for key in ass_loss_classifier_dict}
            total_loss_dict.update(ass_loss_classifier_dict)

            if cfg.TRAIN.LSR_THRES > 0.0:
                lsr_thres = cfg.TRAIN.LSR_THRES
                lsr_loss_S = solver_utils.LSR(F.log_softmax(preds_S, dim=1), dim=1, thres=cfg.TRAIN.LSR_THRES)
                lsr_loss_T = solver_utils.LSR(F.log_softmax(preds_T, dim=1), dim=1, thres=cfg.TRAIN.LSR_THRES)

                total_loss += cfg.TRAIN.LSR_W * lsr_loss_S
                total_loss += cfg.TRAIN.LSR_W * lsr_loss_T

                total_loss_dict['lsr_S'] = lsr_loss_S
                total_loss_dict['lsr_T'] = lsr_loss_T

        preds = F.interpolate(preds_S, size=gt_S.shape[-2:], mode='bilinear', align_corners=False)
        ce_loss = 1.0 * self.CELoss([preds], gt_S)
        if self.distributed:
            lov_loss = lovasz_softmax_multigpu(F.softmax(preds, dim=1), gt_S, classes='present', per_image=False, ignore=255) 
        else:
            lov_loss = lovasz_softmax(F.softmax(preds, dim=1), gt_S, classes='present', per_image=False, ignore=255) 

        ce_loss += (cfg.TRAIN.LOV_W * get_world_size() * self.iter_size) * lov_loss

        total_loss += ce_loss
        total_loss_dict['ce_loss'] = ce_loss
        total_loss_dict['total'] = total_loss

        preds_T = F.interpolate(preds_T, size=gt_S.shape[-2:], mode='bilinear', align_corners=False)
        out_dict = {'feats_S': feats_S, 'feats_T': feats_T, 'preds_S': preds, 'preds_T': preds_T}
        return total_loss_dict, out_dict

    def G_step(self, x_S, x_T):
        self.set_domain_id(1)
        preds_D_T = self.net_D(x_T)

        gt_D_S = to_cuda(torch.FloatTensor(preds_D_T.size()).fill_(1.0))
        loss_D = self.BCELoss(preds_D_T, gt_D_S)

        return loss_D

    def D_step(self, x_S, x_T):
        self.set_domain_id(0)
        preds_D_S = self.net_D(x_S.detach())
        self.set_domain_id(1)
        preds_D_T = self.net_D(x_T.detach())

        preds_D = torch.cat((preds_D_S, preds_D_T), dim=0)

        gt_D_S = to_cuda(torch.FloatTensor(preds_D_S.size()).fill_(1.0))
        gt_D_T = to_cuda(torch.FloatTensor(preds_D_T.size()).fill_(0.0))
        gt_D = torch.cat((gt_D_S, gt_D_T), dim=0)

        loss_D = self.BCELoss(preds_D, gt_D)
        return loss_D

