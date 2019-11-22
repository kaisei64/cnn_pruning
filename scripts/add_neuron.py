import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from dense_mask_generator import DenseMaskGenerator
from dataset import *
from pfgadense import PfgaDense
from dense_evaluateprune import DenseEvaluatePrune
import torch
import torch.nn as nn
import numpy as np
import cloudpickle

# パラメータ利用
with open('./result/CIFAR10_dense_prune.pkl', 'rb') as f:
    new_net = cloudpickle.load(f)
# 全結合層のリスト
dense_list = [new_net.classifier[i] for i in range(len(new_net.classifier)) if isinstance(new_net.classifier[i], nn.Linear)]

# マスクのオブジェクト
de_mask = [DenseMaskGenerator() for _ in range(len(dense_list))]
for i, dense in enumerate(dense_list):
    de_mask[i].mask = np.where(np.abs(dense.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)

ev = [DenseEvaluatePrune() for _ in range(len(dense_list))]
ga = [PfgaDense(dense.out_features, i, evaluate_func=ev[i].evaluate, better_high=True, mutate_rate=0.1)
      for i, dense in enumerate(dense_list)]
best = [list() for _ in range(len(ga))]
max_gen = 50

for i in range(len(ga)):
    while ga[i].generation_num < max_gen:
        ga[i].next_generation()
        best[i] = ga[i].best_gene()
        if best[i] is not None:
            print(f'gen{i + 1}:{ga[i].generation_num} best-value{i + 1}:{best[i][1]}\n')
    # 層ごとに１ニューロンごと追加
    with torch.no_grad():
        add_count = 0
        for j in range(len(dense_list[i].weight.data.cpu().numpy())):
            if np.all(de_mask[i].mask[:, j] == 0):
                de_mask[i].mask[:, j] = 1
                dense_list[i].weight.data[:, j] = torch.tensor(best[i][0], device=device, dtype=dtype)
                add_count += 1
                if add_count == 1:
                    break
    # パラメータの保存
    # with open('./result/CIFAR10_dense_conv_prune.pkl', 'wb') as f:
    #     cloudpickle.dump(new_net, f)
