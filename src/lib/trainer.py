from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from progress.bar import Bar

from model.data_parallel import DataParallel
from utils.utils import AverageMeter

from model.losses import FastFocalLoss, RegWeightedL1Loss
from model.losses import BinRotLoss, WeightedBCELoss
from model.decode import generic_decode
from model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import generic_post_process
from utils.MTL_opt_utils import UncertaintyWeightLoss, GradNormWeightLoss, get_loss_optimizer


class GenericLoss(torch.nn.Module):
    def __init__(self, opt):
        super(GenericLoss, self).__init__()
        self.crit = FastFocalLoss(opt=opt)
        self.crit_reg = RegWeightedL1Loss()
        if 'rot' in opt.heads:
            self.crit_rot = BinRotLoss()
        if 'nuscenes_att' in opt.heads:
            self.crit_nuscenes_att = WeightedBCELoss()
        self.opt = opt

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        if 'hm_hp' in output:
            output['hm_hp'] = _sigmoid(output['hm_hp'])
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        return output

    def forward(self, outputs, batch):
        opt = self.opt
        losses = {head: 0 for head in opt.heads}

        for s in range(opt.num_stacks):
            output = outputs[s]
            output = self._sigmoid_output(output)

            if 'hm' in output:
                losses['hm'] += self.crit(
                    output['hm'], batch['hm'], batch['ind'],
                    batch['mask'], batch['cat']) / opt.num_stacks

            regression_heads = [
                'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
                'dep', 'dim', 'amodel_offset', 'velocity']

            for head in regression_heads:
                if head in output:
                    losses[head] += self.crit_reg(
                        output[head], batch[head + '_mask'],
                        batch['ind'], batch[head]) / opt.num_stacks

            if 'hm_hp' in output:
                losses['hm_hp'] += self.crit(
                    output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
                    batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
                if 'hp_offset' in output:
                    losses['hp_offset'] += self.crit_reg(
                        output['hp_offset'], batch['hp_offset_mask'],
                        batch['hp_ind'], batch['hp_offset']) / opt.num_stacks

            if 'rot' in output:
                losses['rot'] += self.crit_rot(
                    output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
                    batch['rotres']) / opt.num_stacks

            if 'nuscenes_att' in output:
                losses['nuscenes_att'] += self.crit_nuscenes_att(
                    output['nuscenes_att'], batch['nuscenes_att_mask'],
                    batch['ind'], batch['nuscenes_att']) / opt.num_stacks

        losses['tot'] = 0
        for head in opt.heads:
            losses['tot'] += opt.weights[head] * losses[head]

        return losses['tot'], losses


class LossWithStrategy(GenericLoss):
    def __init__(self, opt, logger, param=None):
        super(LossWithStrategy, self).__init__(opt)
        self.weight_strategy = opt.weight_strategy
        if self.weight_strategy == '':
            self.weight = {head: opt.weights[head] for head in opt.heads}
        else:
            self.weight = {head: 1 for head in opt.heads}
        self.logger = logger
        self.groups = opt.groups
        self.group_weight = opt.group_weight
        if self.weight_strategy == 'DWA':
            if opt.resume:
                self.loss_history = opt.his_loss_dict
            else:
                self.loss_history = {group: [] for group in self.groups}
            self.K = len(self.groups)
            self.T = opt.dwa_T
        elif self.weight_strategy == 'UNCER':
            self.group_idx = {group: i for i, group in enumerate(self.groups)}
            self.loss_model = UncertaintyWeightLoss(self.group_idx, self.groups, opt)
            self.optimizer = get_loss_optimizer(model=self.loss_model, opt=opt)
        elif self.weight_strategy == 'GRADNORM':
            self.param = param
            self.group_idx = {group: i for i, group in enumerate(self.groups)}
            self.loss_model = GradNormWeightLoss(self.group_idx, self.groups, opt)
            self.optimizer = get_loss_optimizer(model=self.loss_model, opt=opt)

    def update_weight(self, epoch):
        if self.weight_strategy == "DWA":
            if epoch > 2:
                lambda_w_sum = 0
                lambda_w_group = {group: 0 for group in self.groups}
                for group in self.groups:
                    w = self.loss_history[group][-1] / self.loss_history[group][-2]
                    lambda_w = np.exp(w / self.T)
                    lambda_w_group[group] = lambda_w
                    lambda_w_sum += lambda_w
                for group in self.groups:
                    self.group_weight[group] = self.K * lambda_w_group[group] / lambda_w_sum
                    for head in self.groups[group]:
                        self.weight[head] = self.group_weight[group]
        for group in self.group_weight:
            self.logger.write('{} {:8f} | '.format(group, self.group_weight[group]))
        self.logger.write('\n')
        print(self.weight)

    def update_loss(self, epoch, loss_ret):
        if self.weight_strategy == 'DWA':
            for group in self.groups:
                group_loss = 0
                for head in self.groups[group]:
                    group_loss += loss_ret[head]
                self.loss_history[group].append(group_loss)

    def forward(self, outputs, batch, epoch=None):
        opt = self.opt
        losses = {head: 0 for head in opt.heads}

        for s in range(opt.num_stacks):
            output = outputs[s]
            output = self._sigmoid_output(output)

            if 'hm' in output:
                losses['hm'] += self.crit(
                    output['hm'], batch['hm'], batch['ind'],
                    batch['mask'], batch['cat']) / opt.num_stacks

            regression_heads = [
                'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
                'dep', 'dim', 'amodel_offset', 'velocity']

            for head in regression_heads:
                if head in output:
                    losses[head] += self.crit_reg(
                        output[head], batch[head + '_mask'],
                        batch['ind'], batch[head]) / opt.num_stacks

            if 'hm_hp' in output:
                losses['hm_hp'] += self.crit(
                    output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
                    batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
                if 'hp_offset' in output:
                    losses['hp_offset'] += self.crit_reg(
                        output['hp_offset'], batch['hp_offset_mask'],
                        batch['hp_ind'], batch['hp_offset']) / opt.num_stacks

            if 'rot' in output:
                losses['rot'] += self.crit_rot(
                    output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
                    batch['rotres']) / opt.num_stacks

            if 'nuscenes_att' in output:
                losses['nuscenes_att'] += self.crit_nuscenes_att(
                    output['nuscenes_att'], batch['nuscenes_att_mask'],
                    batch['ind'], batch['nuscenes_att']) / opt.num_stacks

        losses['tot'] = 0
        if self.weight_strategy in ['UNCER', 'GRADNORM']:
            losses['tot'], losses['update'], updated_weight = self.loss_model(losses, self.param, epoch)
            for group in self.group_weight:
                self.group_weight[group] = updated_weight[self.group_idx[group]]
        else:
            for head in opt.heads:
                losses['tot'] += self.weight[head] * losses[head]

        return losses['tot'], losses


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch, epoch=None):
        pre_img = batch['pre_img'] if 'pre_img' in batch else None
        pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
        outputs = self.model(batch['image'], pre_img, pre_hm)
        loss, loss_stats = self.loss(outputs, batch, epoch)
        return outputs[-1], loss, loss_stats


class Trainer(object):
    def __init__(
            self, opt, model, optimizer=None, logger=None):
        self.opt = opt
        self.optimizer = optimizer
        param = list(model.neck.parameters())[-2]
        print(param.shape)
        self.loss_stats, self.loss = self._get_losses(opt, logger, param)
        self.model_with_loss = ModleWithLoss(model, self.loss)
        self.old_norm = 0

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if len(self.opt.gpus) > 1:
            if self.opt.weight_strategy == 'GRADNORM':
                model_with_loss.module.loss.optimizer = get_loss_optimizer(model=model_with_loss.module.loss.loss_model,
                                                                           opt=self.opt)
            model_with_loss.module.loss.update_weight(epoch)
        else:
            if self.opt.weight_strategy == 'GRADNORM':
                model_with_loss.loss.optimizer = get_loss_optimizer(model=model_with_loss.loss.loss_model, opt=self.opt)
            model_with_loss.loss.update_weight(epoch)
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats \
                          if l == 'tot' or opt.weights[l] > 0}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch, epoch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                if opt.weight_strategy == 'GRADNORM':
                    loss.backward(retain_graph=True)
                    if len(self.opt.gpus) > 1:
                        # model_with_loss.module.loss.loss_model.update_weight(model_with_loss.module.model, model_with_loss.module.loss.optimizer, loss_stats)
                        # torch.sum(loss_stats['update']).backward()
                        model_with_loss.module.loss.optimizer.zero_grad()
                        temp_grad = torch.autograd.grad(torch.sum(loss_stats['update']),
                                                        model_with_loss.module.loss.loss_model.weight)[0]
                        grad_norm = torch.norm(temp_grad.data, 1)
                        print(grad_norm)
                        if grad_norm > opt.gradnorm_thred:
                            temp_grad = torch.zeros_like(temp_grad)
                        model_with_loss.module.loss.loss_model.weight.grad = temp_grad
                        model_with_loss.module.loss.optimizer.step()
                    else:
                        # model_with_loss.loss.loss_model.update_weight(model_with_loss.model,
                        #                                               model_with_loss.loss.optimizer, loss_stats)
                        model_with_loss.loss.optimizer.zero_grad()
                        temp_grad = torch.autograd.grad(loss_stats['update'],
                                                        model_with_loss.loss.loss_model.
                                                        weight)[0]
                        grad_norm = torch.norm(temp_grad.data, 1)
                        if grad_norm > opt.gradnorm_thred:
                            temp_grad = torch.zeros_like(temp_grad)
                        model_with_loss.loss.loss_model.weight.grad = temp_grad
                        model_with_loss.loss.optimizer.step()
                else:
                    loss.backward()
                self.optimizer.step()
                if opt.weight_strategy == 'UNCER':
                    if len(self.opt.gpus) > 1:
                        model_with_loss.module.loss.optimizer.step()
                        model_with_loss.module.loss.optimizer.zero_grad()
                        print(model_with_loss.module.loss.group_weight)
                    else:
                        model_with_loss.loss.optimizer.step()
                        model_with_loss.loss.optimizer.zero_grad()
                        print(model_with_loss.loss.group_weight)
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['image'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                      '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:  # If not using progress bar
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.debug > 0:
                self.debug(batch, output, iter_id, dataset=data_loader.dataset)

            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        if len(self.opt.gpus) > 1:
            model_with_loss.module.loss.update_loss(epoch, ret)
        else:
            model_with_loss.loss.update_loss(epoch, ret)
        return ret, results

    def _get_losses(self, opt, logger=None, param=None):
        loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp', \
                      'hp_offset', 'dep', 'dim', 'rot', 'amodel_offset', \
                      'ltrb_amodal', 'tracking', 'nuscenes_att', 'velocity']
        loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]
        # loss = GenericLoss(opt)
        loss = LossWithStrategy(opt, logger, param)
        return loss_states, loss

    def debug(self, batch, output, iter_id, dataset):
        opt = self.opt
        if 'pre_hm' in batch:
            output.update({'pre_hm': batch['pre_hm']})
        dets = generic_decode(output, K=opt.K, opt=opt)
        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()
        dets_gt = batch['meta']['gt_det']
        for i in range(1):
            debugger = Debugger(opt=opt, dataset=dataset)
            img = batch['image'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')

            if 'pre_img' in batch:
                pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
                pre_img = np.clip(((
                                           pre_img * dataset.std + dataset.mean) * 255), 0, 255).astype(np.uint8)
                debugger.add_img(pre_img, 'pre_img_pred')
                debugger.add_img(pre_img, 'pre_img_gt')
                if 'pre_hm' in batch:
                    pre_hm = debugger.gen_colormap(
                        batch['pre_hm'][i].detach().cpu().numpy())
                    debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

            debugger.add_img(img, img_id='out_pred')
            if 'ltrb_amodal' in opt.heads:
                debugger.add_img(img, img_id='out_pred_amodal')
                debugger.add_img(img, img_id='out_gt_amodal')

            # Predictions
            for k in range(len(dets['scores'][i])):
                if dets['scores'][i, k] > opt.vis_thresh:
                    debugger.add_coco_bbox(
                        dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
                        dets['scores'][i, k], img_id='out_pred')

                    if 'ltrb_amodal' in opt.heads:
                        debugger.add_coco_bbox(
                            dets['bboxes_amodal'][i, k] * opt.down_ratio, dets['clses'][i, k],
                            dets['scores'][i, k], img_id='out_pred_amodal')

                    if 'hps' in opt.heads and int(dets['clses'][i, k]) == 0:
                        debugger.add_coco_hp(
                            dets['hps'][i, k] * opt.down_ratio, img_id='out_pred')

                    if 'tracking' in opt.heads:
                        debugger.add_arrow(
                            dets['cts'][i][k] * opt.down_ratio,
                            dets['tracking'][i][k] * opt.down_ratio, img_id='out_pred')
                        debugger.add_arrow(
                            dets['cts'][i][k] * opt.down_ratio,
                            dets['tracking'][i][k] * opt.down_ratio, img_id='pre_img_pred')

            # Ground truth
            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt['scores'][i])):
                if dets_gt['scores'][i][k] > opt.vis_thresh:
                    debugger.add_coco_bbox(
                        dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
                        dets_gt['scores'][i][k], img_id='out_gt')

                    if 'ltrb_amodal' in opt.heads:
                        debugger.add_coco_bbox(
                            dets_gt['bboxes_amodal'][i, k] * opt.down_ratio,
                            dets_gt['clses'][i, k],
                            dets_gt['scores'][i, k], img_id='out_gt_amodal')

                    if 'hps' in opt.heads and \
                            (int(dets['clses'][i, k]) == 0):
                        debugger.add_coco_hp(
                            dets_gt['hps'][i][k] * opt.down_ratio, img_id='out_gt')

                    if 'tracking' in opt.heads:
                        debugger.add_arrow(
                            dets_gt['cts'][i][k] * opt.down_ratio,
                            dets_gt['tracking'][i][k] * opt.down_ratio, img_id='out_gt')
                        debugger.add_arrow(
                            dets_gt['cts'][i][k] * opt.down_ratio,
                            dets_gt['tracking'][i][k] * opt.down_ratio, img_id='pre_img_gt')

            if 'hm_hp' in opt.heads:
                pred = debugger.gen_colormap_hp(
                    output['hm_hp'][i].detach().cpu().numpy())
                gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
                debugger.add_blend_img(img, pred, 'pred_hmhp')
                debugger.add_blend_img(img, gt, 'gt_hmhp')

            if 'rot' in opt.heads and 'dim' in opt.heads and 'dep' in opt.heads:
                dets_gt = {k: dets_gt[k].cpu().numpy() for k in dets_gt}
                calib = batch['meta']['calib'].detach().numpy() \
                    if 'calib' in batch['meta'] else None
                det_pred = generic_post_process(opt, dets,
                                                batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
                                                output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
                                                calib)
                det_gt = generic_post_process(opt, dets_gt,
                                              batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
                                              output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
                                              calib)

                debugger.add_3d_detection(
                    batch['meta']['img_path'][i], batch['meta']['flipped'][i],
                    det_pred[i], calib[i],
                    vis_thresh=opt.vis_thresh, img_id='add_pred')
                debugger.add_3d_detection(
                    batch['meta']['img_path'][i], batch['meta']['flipped'][i],
                    det_gt[i], calib[i],
                    vis_thresh=opt.vis_thresh, img_id='add_gt')
                debugger.add_bird_views(det_pred[i], det_gt[i],
                                        vis_thresh=opt.vis_thresh, img_id='bird_pred_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
