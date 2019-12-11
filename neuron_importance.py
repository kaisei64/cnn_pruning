import torch
import torch.nn as nn
import numpy as np
from result_save_visualization import *

# パラメータ利用
net = parameter_use('./result/CIFAR10_original_train_epoch150.pkl')
dense_list = [net.classifier[i] for i in range(len(net.classifier)) if
              isinstance(net.classifier[i], nn.Linear)]
# L1ノルムのリスト
l1norm_list = [list() for i in range(len(dense_list) - 1)]
for i in range(len(dense_list) - 1):
    for param1, param2 in zip(dense_list[i].weight.data.clone().cpu().numpy(),
                              torch.t(dense_list[i+1].weight.data.clone()).cpu().numpy()):
        l1norm = np.sum(np.abs(param1)) + np.sum(np.abs(param2))
        l1norm_list[i].append(l1norm)


def neuron_importance(dense_num):
    # ニューロン重要度が上位20%と下位5%を保持
    de_high = [list() for _ in range(len(dense_list))]
    de_low = [list() for _ in range(len(dense_list))]

    with torch.no_grad():
        for i in range(len(dense_list) - 1):
            de_high[i] = np.argsort(l1norm_list[i])[:int(dense_list[i].out_features / 20)]
            de_low[i] = np.argsort(l1norm_list[i])[int(dense_list[i].out_features * 99 / 100):]
    return de_high[dense_num], de_low[dense_num]


def neuron_euclidean_distance(ne1, ne2):
    return np.linalg.norm(ne1 - ne2)


def cos_sim(ne1, ne2):
    return np.dot(ne1, ne2) / (np.linalg.norm(ne1) * np.linalg.norm(ne2))
