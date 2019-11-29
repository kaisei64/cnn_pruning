import torch
import torch.nn as nn
import numpy as np
from result_save_visualization import *

# パラメータ利用
net = parameter_use('./result/CIFAR10_original_train.pkl')
# 畳み込み層のリスト
conv_list = [net.features[i] for i in range(len(net.features)) if
             isinstance(net.features[i], nn.Conv2d)]


def channel_importance(conv_num):
    # 勾配と重みの積を保持
    grad_weight_multi_for_each_layer = [conv_list[i].weight.data.cpu().numpy() * conv_list[i].weight.grad.cpu().numpy()
                                        for i in range(len(conv_list))]
    # 勾配と重みの積のL1ノルムを保持
    grad_weight_multi_l1norm = [list() for _ in range(len(conv_list))]
    # チャネル重要度が上位10%と下位10%を保持
    ch_high_10 = [list() for _ in range(len(conv_list))]
    ch_low_5 = [list() for _ in range(len(conv_list))]

    with torch.no_grad():
        for i in range(len(conv_list)):
            for param in grad_weight_multi_for_each_layer[i]:
                grad_weight_multi_l1norm[i].append(np.sum(torch.abs(param).cpu().detach().numpy()))
                ch_high_10[i] = np.argsort(grad_weight_multi_for_each_layer[i])[:int(conv_list[i].out_features / 10)]
                ch_low_5[i] = np.argsort(grad_weight_multi_for_each_layer[i])[int(conv_list[i].out_features * 19 / 20):]

    return ch_high_10[conv_num], ch_low_5[conv_num]


def channel_euclidean_distance(ch1, ch2):
    return np.linalg.norm(ch1 - ch2)
