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
    def __init__(self, head_idx, loss_0, alpha):
        super().__init__()
        self.head_idx = head_idx
        self.num = len(head_idx)
        params = torch.zeros(self.num)
        self.weight = torch.nn.Parameter(params)
        self.loss_0 = loss_0
        self.num = len(self.head_idx)
        self.alpha = alpha
        self.loss_func = nn.L1Loss()
    def forward(self, loss_dict):
        loss_total = 0
        for head in self.head_idx:
            idx = self.head_idx[head]
            loss_total += self.weight[idx] * loss_dict[head]

        return loss_total

    def update_weight(self, MTL_model, loss_optimizer, loss_dict):
        param = list(MTL_model.parameters())
        param_applied = param[0]

        g_total = 0
        l_hat_total = 0

        g_dict = {}
        inv_r_dict = {}
        tar_dict = {}
        l_hat = {}
        for head in self.head_idx:
            gr = torch.autograd.grad(loss_dict[head], param_applied, retain_graph=True, create_graph=True)
            g_dict[head] = torch.norm(gr[0], 2)
            g_total += g_dict[head]
            l_hat[head] = loss_dict[head] / self.loss_0[head]
            l_hat_total += l_hat[head]

        g_ave = g_total / self.num
        l_hat_ave = l_hat_total / self.num

        for head in self.head_idx:
            inv_r_dict[head] = l_hat[head] / l_hat_ave
            tar_dict[head] = (g_ave * (inv_r_dict[head]) ** self.alpha).detach()

        loss_optimizer.zero_grad()

        loss_grad = sum(self.loss_func(g_dict[head], tar_dict[head]) for head in self.head_idx)
        loss_grad.backward()
        loss_optimizer.step()

        weight_total = sum(self.weight[i] for i in range(self.num))
        self.weight /= weight_total

        return







def get_loss_optimizer (model, opt):
    if opt.uncer_optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.uncer_optim == 'sgd':
        print('Using SGD')
        optimizer = torch.optim.SGD(
            model.parameters(), opt.uncer_lr, momentum=0.9, weight_decay=0.0001)
    else:
        assert 0, opt.optim
    return optimizer

