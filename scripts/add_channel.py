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

while ga[0].generation_num < 50:
    ga[0].next_generation()
    best[0] = ga[0].best_gene()
    if best[0] is not None:
        print(f'gen1:{ga[0].generation_num} best-value1:{best[0][1]}\n')
        print(best[0][0])
# 追加
with torch.no_grad():
    add_count = 0
    i = 0
    for j in range(len(conv_list[i].weight.data.cpu().numpy())):
        if np.sum(np.abs(ch_mask[i].mask[j])) < 0.001:
            ch_mask[i].mask[j] = 1
            conv_list[i].weight.data[j] = torch.tensor(best[0][0], device=device, dtype=dtype)
            if i != len(conv_list) - 1:
                ch_mask[i + 1].mask[j, :] = 1
                conv_list[i + 1].weight.data[:, j] = original_conv_list[i + 1].weight.data[:, j].clone()
            add_count += 1
            if add_count == 1:
                print(f'add_filter_conv{i + 1}')
                break
# パラメータの保存
with open('./result/CIFAR10_dense_conv_prune.pkl', 'wb') as f:
    cloudpickle.dump(new_net, f)

# while ga[1].generation_num < 50:
#     ga[1].next_generation()
#     best[1] = ga[1].best_gene()
#     if best[1] is not None:
#         print('gen2:{} best-value2:{}'.format(ga[1].generation_num, best[1][1]))

    # ga[2].next_generation()
    # best3 = ga[2].best_gene()
    # if best3 is not None:
    #     print('gen3:{} best-value3:{}'.format(ga[2].generation_num, best3[1]))

    # ga[3].next_generation()
    # best4 = ga[3].best_gene()
    # if best4 is not None:
    #     print('gen4:{} best-value4:{}'.format(ga[3].generation_num, best4[1]))

    # ga[4].next_generation()
    # best5 = ga[4].best_gene()
    # if best5 is not None:
    #     print('gen5:{} best-value5:{}'.format(ga[4].generation_num, best5[1]))
