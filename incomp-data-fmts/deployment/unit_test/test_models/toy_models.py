import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyFPConv(nn.Module):

    def __init__(self, n=2):
        super().__init__()
        self.input_shape = (1, 32, 32)
        self.net = nn.ModuleDict(
            {f'conv{idx}': nn.Conv2d(idx, idx+1, 3, padding=1)
             for idx in range(1, n+1)}
        )

    def forward(self, x):
        for idx, conv in enumerate(self.net.values()):
            if idx == 0:
                x = conv(x)
            else:
                x = conv(F.relu(x))
        return x


class ToyQConv(nn.Module):

    def __init__(self, conv_func, abits, wbits):
        super().__init__()
        self.input_shape = (1, 32, 32)
        self.conv_func = conv_func
        self.abits = abits
        self.wbits = wbits

        self.conv0 = conv_func(1, 3, abits=abits[0], wbits=wbits[0],
                               kernel_size=3, stride=1, bias=True,
                               padding=1, groups=1, first_layer=False)
        self.conv1 = conv_func(3, 5, abits=abits[0], wbits=wbits[0],
                               kernel_size=3, stride=1, bias=True,
                               padding=1, groups=1)
        self.fc = conv_func(5, 2, abits=abits[0], wbits=wbits[0],
                            kernel_size=32, stride=1, bias=True,
                            padding=0, groups=1, fc='multi')

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.fc(x1)
        return x2[:, :, 0, 0]

    def harden_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, self.conv_func):
                module.harden_weights()

    def store_hardened_weights(self):
        with torch.no_grad():
            for _, module in self.named_modules():
                if isinstance(module, self.conv_func):
                    module.store_hardened_weights()
