import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from dense_mask_generator import DenseMaskGenerator
from dataset import *
from pfgadense import PfgaDense
from dense_evaluateprune import DenseEvaluatePrune
from result_save_visualization import *
import torch
import torch.nn as nn
import numpy as np

# パラメータ利用
new_net = parameter_use('./result/CIFAR10_dense_conv_prune.pkl')
# 全結合層のリスト
dense_list = [new_net.classifier[i] for i in range(len(new_net.classifier)) if isinstance(new_net.classifier[i], nn.Linear)]

# マスクのオブジェクト
de_mask = [DenseMaskGenerator() for _ in range(len(dense_list))]
for i, dense in enumerate(dense_list):
    de_mask[i].mask = np.where(np.abs(dense.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)

gen_num = 10
add_neuron_num = 1

# 追加前重み分布の描画
for i in range(len(dense_list)):
    before_weight = [np.sum(dense_list[i].weight.data[k].cpu().detach().numpy()) for k
                     in range(len(dense_list[i].weight.data.cpu().numpy()))]
    parameter_distribution_vis(f'./figure/dis_vis/dense{i + 1}/before_weight_distribution{i + 1}.png', before_weight)

for count in range(add_neuron_num):
    ev = [DenseEvaluatePrune() for _ in range(len(dense_list))]
    ga = [PfgaDense(dense.out_features, i, evaluate_func=ev[i].evaluate, better_high=False, mutate_rate=0.1)
          for i, dense in enumerate(dense_list)]
    best = [list() for _ in range(len(ga))]
    for i in range(len(ga)):
        while ga[i].generation_num < gen_num:
            ga[i].next_generation()
            best[i] = ga[i].best_gene()
            if best[i] is not None:
                print(f'gen{i + 1}:{ga[i].generation_num} best-value{i + 1}:{best[i][1]}\n')
        # 層ごとに１ニューロンごと追加
        with torch.no_grad():
            for j in range(len(dense_list[i].weight.data.cpu().numpy())):
                if np.all(de_mask[i].mask[:, j] == 0):
                    de_mask[i].mask[:, j] = 1
                    dense_list[i].weight.data[:, j] = torch.tensor(best[i][0], device=device, dtype=dtype)
                    break

        # 追加後重み分布の描画
        after_weight = [np.sum(dense_list[i].weight.data[k].cpu().numpy()) for k
                        in range(len(dense_list[i].weight.data.cpu().numpy()))]
        parameter_distribution_vis(f'./figure/dis_vis/dense{i + 1}/after{count + 1}_weight_distribution{i + 1}.png',
                                   after_weight)

    # パラメータの保存
    # parameter_save('./result/CIFAR10_dense_conv_prune.pkl', new_net)
