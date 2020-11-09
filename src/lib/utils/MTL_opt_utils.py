import torch
import torch.nn as nn


class UncertaintyWeightLoss(nn.Module):
    def __init__(self, head_idx, method):
        super(UncertaintyWeightLoss, self).__init__()
        self.head_idx = head_idx
        self.num = len(head_idx)
        params = torch.zeros(self.num)
        self.log_sigma = torch.nn.Parameter(params)
        self.method = method

    def forward(self, loss_dict):
        loss_total = 0
        for head in self.head_idx:
            idx = self.head_idx[head]
            if self.method == "BASIC":
                loss_total += 1 / (2 * torch.exp(2 * self.log_sigma[idx]) * loss_dict[head]) + self.log_sigma[idx]
            else:
                loss_total += 1 / (2 * torch.exp(2 * self.log_sigma[idx]) * loss_dict[head]) + torch.log(
                    torch.exp(2 * self.log_sigma[idx]) + 1)
        return loss_total


class GradNormWeightLoss(nn.Module):
    def __init__(self, head_idx, alpha):
        super().__init__()
        self.head_idx = head_idx
        self.num = len(head_idx)
        params = torch.ones(self.num, requires_grad=True)
        self.weight = torch.nn.Parameter(params)
        self.loss_0 = {}
        self.num = len(self.head_idx)
        self.alpha = alpha
        self.loss_func = nn.L1Loss()
        self.weighted_loss = {}

    def forward(self, loss_dict):
        loss_total = 0
        for head in self.head_idx:
            idx = self.head_idx[head]
            self.weighted_loss[head] = self.weight[idx] * loss_dict[head]
            loss_total += self.weighted_loss[head]

        return loss_total

    def update_weight(self, MTL_model, loss_optimizer, loss_dict):
        if len(self.loss_0) == 0:
            for head in self.head_idx:
                self.loss_0[head] = loss_dict[head].data

        param = list(MTL_model.ida_up.parameters())
        g_total = 0
        l_hat_total = 0

        g_dict = {}
        inv_r_dict = {}
        tar_dict = {}
        l_hat = {}
        for head in self.head_idx:
            idx = self.head_idx[head]
            gr = torch.autograd.grad(self.weighted_loss[head], param[17], retain_graph=True, create_graph=True)
            g_dict[head] = torch.norm(gr[0], 2)
            g_total += g_dict[head]
            l_hat[head] = self.weighted_loss[head] / self.loss_0[head]
            l_hat_total += l_hat[head]

        g_ave = g_total / self.num
        l_hat_ave = l_hat_total / self.num

        for head in self.head_idx:
            inv_r_dict[head] = l_hat[head] / l_hat_ave
            tar_dict[head] = (g_ave * (inv_r_dict[head]) ** self.alpha).detach()

        loss_optimizer.zero_grad()

        loss_grad = sum(self.loss_func(g_dict[head], tar_dict[head]) for head in self.head_idx)
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

