import torch
import torch.nn as nn


class UncertaintyWeightLoss(nn.Module):
    def __init__(self, group_idx, groups, method):
        super(UncertaintyWeightLoss, self).__init__()
        self.group_idx = group_idx
        self.groups = groups
        self.num = len(group_idx)
        params = torch.zeros(self.num)
        self.log_sigma = torch.nn.Parameter(params)
        self.method = method

    def forward(self, loss_dict):
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
        return loss_total


class GradNormWeightLoss(nn.Module):
    def __init__(self, group_idx, groups, alpha):
        super().__init__()
        self.group_idx = group_idx
        self.groups = groups
        self.num = len(group_idx)
        params = torch.ones(self.num, requires_grad=True)
        self.weight = torch.nn.Parameter(params)
        self.loss_0 = {}
        self.num = len(self.group_idx)
        self.alpha = alpha
        self.loss_func = nn.L1Loss()
        self.weighted_loss = {}

    def forward(self, loss_dict):
        loss_total = 0
        for group in self.group_idx:
            idx = self.group_idx[group]
            group_loss = 0
            for head in self.groups[group]:
                group_loss += loss_dict[head]
            self.weighted_loss[group] = self.weight[idx] * group_loss
            loss_total += self.weighted_loss[group]

        return loss_total

    def update_weight(self, MTL_model, loss_optimizer, loss_dict):
        if len(self.loss_0) == 0:
            for group in self.group_idx:
                group_l = 0
                for head in self.groups[group]:
                    group_l += loss_dict[head].data
                self.loss_0[group] = group_l

        param = list(MTL_model.ida_up.parameters())
        g_total = 0
        l_hat_total = 0

        g_dict = {}
        inv_r_dict = {}
        tar_dict = {}
        l_hat = {}
        for group in self.group_idx:
            gr = torch.autograd.grad(self.weighted_loss[group], param[17], retain_graph=True, create_graph=True)
            g_dict[group] = torch.norm(gr[0], 2)
            g_total += g_dict[group]
            l_hat[group] = self.weighted_loss[group] / self.loss_0[group]
            l_hat_total += l_hat[group]

        g_ave = g_total / self.num
        l_hat_ave = l_hat_total / self.num

        for group in self.group_idx:
            inv_r_dict[group] = l_hat[group] / l_hat_ave
            tar_dict[group] = (g_ave * (inv_r_dict[group]) ** self.alpha).detach()

        loss_optimizer.zero_grad()

        loss_grad = sum(self.loss_func(g_dict[group], tar_dict[group]) for group in self.group_idx)
        # print(loss_grad)
        loss_grad.backward()
        loss_optimizer.step()

        weight_total = sum(self.weight[i] for i in range(self.num))
        # print(self.weight.grad)
        torch.div(self.weight, weight_total)
        # print(self.weight)
        return







def get_loss_optimizer(model, opt):
    if opt.weight_optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.weight_optim == 'sgd':
        print('Using SGD')
        optimizer = torch.optim.SGD(
            model.parameters(), opt.weight_optim_lr, momentum=0.9, weight_decay=0.0001)
    else:
        assert 0, opt.optim
    return optimizer

