import torch
import torch.nn as nn
# import torch.nn.functional as F


# class Conv2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
#                  padding_mode='zeros'):
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
#         self.mask = None
#
#     def forward(self, input, flag, mask):
#         self.mask = mask
#         if not flag:
#             if self.padding_mode == 'circular':
#                 expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
#                                     (self.padding[0] + 1) // 2, self.padding[0] // 2)
#                 return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
#                                 self.weight, self.bias, self.stride,
#                                 _pair(0), self.dilation, self.groups)
#             return F.conv2d(input, self.weight, self.bias, self.stride,
#                             self.padding, self.dilation, self.groups)
#         else:
#             mask = torch.tensor(self.mask, device=device, dtype=dtype)
#             with torch.no_grad():
#                 self.weight *= mask
#             if self.padding_mode == 'circular':
#                 expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
#                                     (self.padding[0] + 1) // 2, self.padding[0] // 2)
#                 return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
#                                 self.weight, self.bias, self.stride,
#                                 _pair(0), self.dilation, self.groups)
#             return F.conv2d(input, self.weight, self.bias, self.stride,
#                             self.padding, self.dilation, self.groups)
#
#
# class Mylinear(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True):
#         super().__init__(in_features, out_features, bias)
#         self.mask = None
#
#     def forward(self, input, flag, mask):
#         self.mask = mask
#         if not flag:
#             return F.linear(input, self.weight, self.bias)
#         else:
#             # print(mask.shape)
#             # print(self.weight.shape)
#             # print(self.bias.shape)
#             # print(input.shape)
#             mask = torch.t(torch.tensor(self.mask, device=device, dtype=dtype))
#             return F.linear(input, self.weight * mask, self.bias)


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv1_coef = torch.ones(64)
        self.conv2_coef = torch.ones(192)
        self.conv3_coef = torch.ones(384)
        self.conv4_coef = torch.ones(256)
        self.conv5_coef = torch.ones(256)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, num_classes)
        self.drop = nn.Dropout()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, flag, coef):
        x = self.pool(self.activation(self.conv1(x) * self.conv1_coef))
        x = self.pool(self.activation(self.conv2(x) * self.conv2_coef))
        x = self.activation(self.conv3(x) * self.conv3_coef)
        x = self.activation(self.conv4(x) * self.conv4_coef)
        x = self.pool(self.activation(self.conv5(x) * self.conv5_coef))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(self.drop(x)))
        x = self.activation(self.fc2(self.drop(x)))
        x = self.fc3(x)
        return x
