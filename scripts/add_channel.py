import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from pfgacnn import PfgaCnn
from evaluateprune import EvaluatePrune
import torch.nn as nn
import cloudpickle

# パラメータ利用
with open('CIFAR10_conv_prune.pkl', 'rb') as f:
    new_net = cloudpickle.load(f)
# 畳み込み層のリスト
conv_list = [new_net.features[i] for i in range(len(new_net.features)) if isinstance(new_net.features[i], nn.Conv2d)]

ev = EvaluatePrune()
ga = [PfgaCnn(conv.in_channels, conv.kernel_size, i, evaluate_func=ev.evaluate, better_high=True, mutate_rate=0.1)
      for i, conv in enumerate(conv_list)]

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

    # ga[3].next_generation()
    # best4 = ga[3].best_gene()
    # if best4 is not None:
    #     print('gen4:{} best-value4:{}'.format(ga[3].generation_num, best4[1]))

    # ga[4].next_generation()
    # best5 = ga[4].best_gene()
    # if best5 is not None:
    #     print('gen5:{} best-value5:{}'.format(ga[4].generation_num, best5[1]))
