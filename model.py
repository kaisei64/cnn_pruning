import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 10

device = 'cuda'
dtype = torch.float


class MyConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.mask = None

    def forward(self, input, flag, mask):
        self.mask = mask
        if not flag:
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                    (self.padding[0] + 1) // 2, self.padding[0] // 2)
                return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                self.weight, self.bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            mask = torch.tensor(self.mask, device=device, dtype=dtype)
            with torch.no_grad():
                self.weight *= mask
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                    (self.padding[0] + 1) // 2, self.padding[0] // 2)
                return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                self.weight, self.bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


class Mylinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.mask = None

    def forward(self, input, flag, mask):
        self.mask = mask
        if not flag:
            return F.linear(input, self.weight, self.bias)
        else:
            # print(mask.shape)
            # print(self.weight.shape)
            # print(self.bias.shape)
            # print(input.shape)
            mask = torch.t(torch.tensor(self.mask, device=device, dtype=dtype))
            return F.linear(input, self.weight * mask, self.bias)


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = MyConv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.batchnorm1 = nn.BatchNorm2d(64)
        # self.batchnorm2 = nn.BatchNorm2d(192)
        # self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv2 = MyConv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = MyConv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = MyConv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = MyConv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = Mylinear(256, 100)
        self.fc2 = Mylinear(100, num_classes)

    def forward(self, x, flag_c, flag_f, mask1, mask2, mask3, mask4, mask5, mask6, mask7):
        x = self.pool(F.relu(self.conv1(x, flag_c, mask1)))
        # x = self.batchnorm1(self.pool(F.relu(self.conv1(x, flag_c, mask1))))
        x = self.pool(F.relu(self.conv2(x, flag_c, mask2)))
        # x = self.batchnorm2(self.pool(F.relu(self.conv2(x, flag_c, mask2))))
        x = self.pool(F.relu(self.conv3(x, flag_c, mask3)))
        x = self.pool(F.relu(self.conv4(x, flag_c, mask4)))
        x = self.pool(F.relu(self.conv5(x, flag_c, mask5)))
        # x = self.batchnorm5(self.pool(F.relu(self.conv5(x, flag_c, mask5))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x, flag_f, mask6))
        x = self.fc2(x, flag_f, mask7)
        return x
