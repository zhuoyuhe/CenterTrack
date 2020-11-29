import torch
import torch.nn as nn
import time

class UncertaintyWeightLoss(nn.Module):
    def __init__(self, group_idx, groups, opt):
        super(UncertaintyWeightLoss, self).__init__()
        self.group_idx = group_idx
        self.groups = groups
        self.num = len(group_idx)
        params = torch.zeros(self.num)
        if opt.resume:
            for group in groups:
                params[group_idx[group]] = opt.logsigma[group]
        self.log_sigma = torch.nn.Parameter(params)
        self.method = opt.uncer_mode

    def forward(self, loss_dict, param=None, epoch=None):
        loss_total = 0
        for group in self.group_idx:
            idx = self.group_idx[group]
            group_loss = 0
            for head in self.groups[group]:
                group_loss +=loss_dict[head]
            if self.method == "BASIC":
                loss_total += 1 / (2 * torch.exp(2 * self.log_sigma[idx])) * group_loss + self.log_sigma[idx]
            else:
                loss_total += 1 / (2 * torch.exp(2 * self.log_sigma[idx])) * group_loss + torch.log(
                    torch.exp(2 * self.log_sigma[idx]) + 1)
        return loss_total, 0, self.log_sigma


class GradNormWeightLoss(nn.Module):
    def __init__(self, group_idx, groups, opt):
        super().__init__()
        self.group_idx = group_idx
        self.groups = groups
        self.num = len(group_idx)
        params = torch.ones(self.num, requires_grad=True)
        if opt.resume:
            for group in groups:
                params[group_idx[group]] = opt.grad_weight[group]
        self.weight = torch.nn.Parameter(params)
        self.loss_0 = {}
        self.num = len(self.group_idx)
        self.alpha = opt.gradnorm_alpha
        self.loss_func = nn.L1Loss()
        self.weighted_loss = {}
        if opt.resume:
            for group in groups:
                self.loss_0[group] = opt.grad_l0[group]
            print(self.loss_0)

    def forward(self, loss_dict, param, epoch):
        weight_total = sum(self.weight[i] for i in range(self.num))
        # print(self.weight.grad)
        self.weight.data = (self.num * self.weight/weight_total)
        print(self.weight)
        loss_total = 0
        for group in self.group_idx:
            idx = self.group_idx[group]
            group_loss = 0
            for head in self.groups[group]:
                group_loss += loss_dict[head]
            self.weighted_loss[group] = self.weight[idx] * group_loss
            loss_total += self.weighted_loss[group]

        if len(self.loss_0) == 0 or epoch == 1:
            for group in self.group_idx:
                group_l = 0
                for head in self.groups[group]:
                    group_l += loss_dict[head].data
                self.loss_0[group] = group_l


        g_total = 0
        l_hat_total = 0

        g_dict = {}
        inv_r_dict = {}
        tar_dict = {}
        l_hat = {}
        for group in self.group_idx:
            group_loss = 0
            for head in self.groups[group]:
                group_loss += loss_dict[head]
            group_loss = torch.sum(group_loss)
            gr = torch.autograd.grad(group_loss, param, retain_graph=True, create_graph=True)[0]
            gr = self.weight[self.group_idx[group]] * gr
            g_dict[group] = torch.norm(gr, 2)
            # print(torch.autograd.grad(g_dict[group], self.weight, retain_graph=True, create_graph=True))
            g_total += g_dict[group]
            l_hat[group] = group_loss / self.loss_0[group]
            l_hat_total += l_hat[group]

        g_ave = g_total / self.num
        l_hat_ave = l_hat_total / self.num

        for group in self.group_idx:
            inv_r_dict[group] = l_hat[group] / l_hat_ave
            tar_dict[group] = (g_ave * (inv_r_dict[group]) ** self.alpha).detach()

        loss_grad = sum(self.loss_func(g_dict[group], tar_dict[group]) for group in self.group_idx)
        # print(loss_grad)
        # loss_grad.backward()
        # print(self.weight)
        return loss_total, loss_grad, self.weight







def get_loss_optimizer(model, opt):
    if opt.weight_optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.weight_optim == 'sgd':
        print('Using SGD')
        optimizer = torch.optim.SGD(
            model.parameters(), opt.weight_optim_lr, momentum=0, weight_decay=0.0)
    else:
        assert 0, opt.optim
    return optimizer

