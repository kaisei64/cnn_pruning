import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from channel_mask_generator import ChannelMaskGenerator
from dataset import *
from pfgacnn import PfgaCnn
from cnn_evaluateprune import CnnEvaluatePrune
import torch
import torch.nn as nn
import numpy as np
import cloudpickle

# 枝刈り前パラメータ利用
with open('./result/CIFAR10_original_train.pkl', 'rb') as f:
    original_net = cloudpickle.load(f)
# 畳み込み層のリスト
original_conv_list = [original_net.features[i] for i in range(len(original_net.features)) if
                      isinstance(original_net.features[i], nn.Conv2d)]

# 枝刈り後パラメータ利用
with open('./result/CIFAR10_dense_conv_prune.pkl', 'rb') as f:
    new_net = cloudpickle.load(f)
# 畳み込み層のリスト
conv_list = [new_net.features[i] for i in range(len(new_net.features)) if isinstance(new_net.features[i], nn.Conv2d)]
# マスクのオブジェクト
ch_mask = [ChannelMaskGenerator() for _ in range(len(conv_list))]
for i, conv in enumerate(conv_list):
    ch_mask[i].mask = np.where(np.abs(conv.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)

ev = CnnEvaluatePrune()
ga = [PfgaCnn(conv.in_channels, conv.kernel_size, i, evaluate_func=ev.evaluate, better_high=True, mutate_rate=0.1)
      for i, conv in enumerate(conv_list)]
best = [list() for _ in range(len(ga))]
max_gen = 50

for i in range(len(ga)):
    while ga[i].generation_num < max_gen:
        ga[i].next_generation()
        best[i] = ga[i].best_gene()
        if best[i] is not None:
            print(f'gen1:{ga[i].generation_num} best-value1:{best[i][1]}\n')
    # 層ごとに１チャネルごと追加
    with torch.no_grad():
        add_count = 0
        for j in range(len(conv_list[i].weight.data.cpu().numpy())):
            if np.sum(np.abs(ch_mask[i].mask[j])) < 0.001:
                ch_mask[i].mask[j] = 1
                conv_list[i].weight.data[j] = torch.tensor(best[i][0], device=device, dtype=dtype)
                if i != len(conv_list) - 1:
                    ch_mask[i + 1].mask[j, :] = 1
                    conv_list[i + 1].weight.data[:, j] = original_conv_list[i + 1].weight.data[:, j].clone()
                add_count += 1
                if add_count == 1:
                    break
    # パラメータの保存
    # with open('./result/CIFAR10_dense_conv_prune.pkl', 'wb') as f:
    #     cloudpickle.dump(new_net, f)
