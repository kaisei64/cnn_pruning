import torch
import torch.nn as nn
import numpy as np
from result_save_visualization import *

# パラメータ利用
net = parameter_use('./result/CIFAR10_original_train_epoch150.pkl')
dense_list = [net.classifier[i] for i in range(len(net.classifier)) if
              isinstance(net.classifier[i], nn.Linear)]
# L1ノルムのリスト
l1norm_list = list()
for i, dense in enumerate(dense_list):
    for param in dense:
        for j in param:
            l1norm_list[i].append(np.sum(np.abs(j)))


def neuron_importance(dense_num):
    # ニューロン重要度が上位20%と下位5%を保持
    de_high = [list() for _ in range(len(dense_list))]
    de_low = [list() for _ in range(len(dense_list))]

    with torch.no_grad():
        for i in range(len(dense_list)):
            de_high[i] = np.argsort(l1norm_list[i])[:int(dense_list[i].out_features / 4)]
            de_low[i] = np.argsort(l1norm_list[i])[int(dense_list[i].out_features * 19 / 20):]
    return de_high[dense_num], de_low[dense_num]


def neuron_euclidean_distance(ne1, ne2):
    return pow(np.linalg.norm(ne1 - ne2), 2)


def cos_sim(ne1, ne2):
    return np.dot(ne1, ne2) / (np.linalg.norm(ne1) * np.linalg.norm(ne2))
