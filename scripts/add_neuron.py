import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from pfgadense import PfgaDense
from dense_evaluateprune import DenseEvaluatePrune
import torch.nn as nn
import cloudpickle

# パラメータ利用
with open('./result/CIFAR10_dense_prune.pkl', 'rb') as f:
    new_net = cloudpickle.load(f)
# 畳み込み層のリスト
dense_list = [new_net.classifier[i] for i in range(len(new_net.classifier)) if isinstance(new_net.classifier[i], nn.Linear)]

ev = DenseEvaluatePrune()
ga = [PfgaDense(dense.in_features, dense.out_features, i, evaluate_func=ev.evaluate, better_high=True, mutate_rate=0.1)
      for i, dense in enumerate(dense_list)]

while True:
    ga[0].next_generation()
    best1 = ga[0].best_gene()
    if best1 is not None:
        print('gen1:{} best-value1:{}'.format(ga[0].generation_num, best1[1]))

    # ga[1].next_generation()
    # best2 = ga[1].best_gene()
    # if best2 is not None:
    #     print('gen2:{} best-value2:{}'.format(ga[1].generation_num, best2[1]))

    # ga[2].next_generation()
    # best3 = ga[2].best_gene()
    # if best3 is not None:
    #     print('gen3:{} best-value3:{}'.format(ga[2].generation_num, best3[1]))
