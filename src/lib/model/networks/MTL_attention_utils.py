import torch
import torch.nn as nn
from torch.autograd import Variable

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PadNet(nn.Module):
    def __init__(self, opt, head_convs, head_kernel=3):
        super(PadNet, self).__init__()
        self.heads = opt.heads
        self.opt = opt
        for head in self.heads:
            num_class = self.heads[head]
            Y2F = nn.Sequential(nn.Conv2d(num_class, opt.pad_channel, stride=1, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))
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
            if self.opt.pad_grouping:
                for d_head in self.opt.pad_group[head]:
                    distill_f[head] += torch.mul(gate[head], weighted_f[d_head])
            else:
                for d_head in self.heads:
                    if d_head != head:
                        distill_f[head] += torch.mul(gate[head], weighted_f[d_head])

            output[head] = self.__getattr__(head + 'out')(distill_f[head])

        return output

class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvGRUCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wir = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whr = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wiz = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whz = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Win = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whn = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.br = None
        self.bz = None
        self.bin = None
        self.bhn = None

    def forward(self, x, h):
        rt = torch.sigmoid(self.Wir(x) + self.Whr(h) + self.br) #reset
        zt = torch.sigmoid(self.Wiz(x) + self.Whz(h) + self.bz) #update
        nt = torch.tanh(self.Win(x) + self.bin + rt * (self.Whn(h) + self.bhn)) #new
        ht = (1-zt) * nt + zt * h
        return ht

    def init_hidden(self, batch_size, hidden, shape):
        if self.br is None:
            self.br = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.bz = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.bin = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.bhn = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.br.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.br.size()[3], 'Input Width Mismatched!'
        return Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda()

class FADNet(nn.Module):
    def __init__(self,opt):
        super(FADNet, self).__init__()
        self.opt = opt
        self.input_channels = [opt.input_channels] + opt.hidden_channels
        self.hidden_channels = opt.hidden_channels
        self.kernel_size = opt.fad_kernel_size
        self.num_layers = len(opt.hidden_channels)
        self.step = opt.fad_step
        if len (opt.effective_step) != 0:
            self.effective_step = opt.effective_step
        else:
            self.effective_step = [i for i in range(self.step)]
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvGRUCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    h = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append(h)

                # do forward
                h = internal_state[i]
                x = getattr(self, name)(x, h)
                internal_state[i] = x
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)


        return outputs
