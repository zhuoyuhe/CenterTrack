import torch
import torch.nn as nn

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PadNet(nn.Module):
    def __init__(self, opt, head_convs, head_kernel=3):
        super(PadNet, self).__init__()
        self.heads = opt.heads
        for head in self.heads:
            num_class = self.heads[head]
            Y2F = nn.Sequential(nn.Conv2d(num_class, opt.pad_channel, stride=1, kernel_size=3, padding=1),
                                nn.ReLU())
            get_gate = nn.Sequential(nn.Conv2d(opt.pad_channel, opt.pad_channel, stride=1, kernel_size=3, padding=1),
                                 nn.Sigmoid())
            trans = nn.Conv2d(opt.pad_channel, opt.pad_channel, stride=1, kernel_size=3, padding=1)
            self.__setattr__(head + 'Y2F', Y2F)
            self.__setattr__(head + 'gate', get_gate)
            self.__setattr__(head + 'trans', trans)

            head_conv = head_convs[head]
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], num_class,
                    kernel_size=1, stride=1, padding=0, bias=True)
              conv = nn.Conv2d(opt.pad_channel, head_conv[0],
                               kernel_size=head_kernel,
                               padding=head_kernel // 2, bias=True)
              convs = [conv]
              for k in range(1, len(head_conv)):
                  convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k],
                               kernel_size=1, bias=True))
              if len(convs) == 1:
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
              elif len(convs) == 2:
                fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True),
                  convs[1], nn.ReLU(inplace=True), out)
              elif len(convs) == 3:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True),
                    convs[1], nn.ReLU(inplace=True),
                    convs[2], nn.ReLU(inplace=True), out)
              elif len(convs) == 4:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True),
                    convs[1], nn.ReLU(inplace=True),
                    convs[2], nn.ReLU(inplace=True),
                    convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(opt.pad_channel, num_class,
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head + 'out', fc)
        a=1

    def forward(self, inter_out):
        f = {}
        gate = {}
        weighted_f = {}
        distill_f = {}
        output = {}
        for head in self.heads:
            f[head] = self.__getattr__(head + 'Y2F')(inter_out[head])
            gate[head] = self.__getattr__(head + 'gate')(f[head])
            weighted_f[head] = self.__getattr__(head + 'trans')(f[head])
            distill_f[head] = f[head].clone()
        for head in self.heads:
            for d_head in self.heads:
                if d_head != head:
                    distill_f[head] += torch.mul(gate[head], weighted_f[d_head])

            output[head] = self.__getattr__(head + 'out')(distill_f[head])

        return output
