import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from channel_mask_generator import ChannelMaskGenerator
from dataset import *
from pfgacnn import PfgaCnn
from cnn_evaluateprune import CnnEvaluatePrune
from result_save_visualization import *
import torch
import torch.nn as nn
import numpy as np

# 枝刈り前パラメータ利用
original_net = parameter_use('./result/CIFAR10_original_train.pkl')
# 枝刈り前畳み込み層のリスト
original_conv_list = [original_net.features[i] for i in range(len(original_net.features)) if
                      isinstance(original_net.features[i], nn.Conv2d)]
# 枝刈り後パラメータ利用
new_net = parameter_use('./result/CIFAR10_dense_conv_prune.pkl')
# 枝刈り後畳み込み層のリスト
conv_list = [new_net.features[i] for i in range(len(new_net.features)) if isinstance(new_net.features[i], nn.Conv2d)]
# マスクのオブジェクト
ch_mask = [ChannelMaskGenerator() for _ in range(len(conv_list))]
for i, conv in enumerate(conv_list):
    ch_mask[i].mask = np.where(np.abs(conv.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)

gen_num = 30
add_channel_num = 10

# 追加前重み分布の描画
for i in range(len(conv_list)):
    before_weight = [np.sum(conv_list[i].weight.data[k].cpu().detach().numpy()) for k
                     in range(len(conv_list[i].weight.data.cpu().numpy()))]
    parameter_distribution_vis(f'./figure/dis_vis/conv{i + 1}/before_weight_distribution{i + 1}.png', before_weight)

for count in range(add_channel_num):
    ev = [CnnEvaluatePrune() for _ in range(len(conv_list))]
    ga = [PfgaCnn(conv.in_channels, conv.kernel_size, i,
                  evaluate_func=ev[i].evaluate, better_high=True, mutate_rate=0.1) for i, conv in enumerate(conv_list)]
    best = [list() for _ in range(len(ga))]
    for i in range(len(ga)):
        while ga[i].generation_num < gen_num:
            ga[i].next_generation()
            best[i] = ga[i].best_gene()
            if best[i] is not None:
                print(f'gen{i + 1}:{ga[i].generation_num} best-value{i + 1}:{best[i][1]}\n')

        with torch.no_grad():
            # 層ごとに１チャネルごと追加
            add_count = 0
            for j in range(len(conv_list[i].weight.data.cpu().numpy())):
                if np.sum(np.abs(ch_mask[i].mask[j])) < 25 * (count + 1) + 1:
                    ch_mask[i].mask[j] = 1
                    conv_list[i].weight.data[j] = torch.tensor(best[i][0], device=device, dtype=dtype)
                    if i != len(conv_list) - 1:
                        ch_mask[i + 1].mask[j, :] = 1
                        conv_list[i + 1].weight.data[:, j] = original_conv_list[i + 1].weight.data[:, j].clone()
                    add_count += 1
                    if add_count == 1:
                        break

            # 追加後重み分布の描画
            after_weight = [np.sum(conv_list[i].weight.data[k].cpu().numpy()) for k
                            in range(len(conv_list[i].weight.data.cpu().numpy()))]
            parameter_distribution_vis(f'./figure/dis_vis/conv{i + 1}/after{count + 1}_weight_distribution{i + 1}.png', after_weight)

            # 追加後チャネル可視化
            for j in range(conv_list[i].out_channels):
                for k in range(len(conv_list[i].weight.data.cpu().numpy())):
                    if np.sum(np.abs(ch_mask[i].mask[k])) > 25 * (count + 1) + 1:
                        print(np.sum(np.abs(ch_mask[i].mask[k])))
                        conv_vis(f'./figure/ch_vis/conv{i + 1}/after{count + 1}_conv{i + 1}_filter{j + 1}.png'
                                 , conv_list[i].weight.data.cpu().numpy(), j)

        # パラメータの保存
        parameter_save('./result/CIFAR10_dense_conv_prune.pkl', new_net)
