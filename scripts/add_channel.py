import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from pfga_cnn import PFGA
from evaluateprune import EvaluatePrune
import torch.nn as nn
import cloudpickle

# パラメータ利用
with open('CIFAR10_conv_prune.pkl', 'rb') as f:
    new_net = cloudpickle.load(f)
# 畳み込み層のリスト
conv_list = [new_net.features[i] for i in range(len(new_net.features)) if isinstance(new_net.features[i], nn.Conv2d)]

ev = EvaluatePrune()
ga = [PFGA(conv.in_channels, conv.kernel_size, evaluate_func=ev.evaluate, better_high=True, mutate_rate=0.1) for conv in conv_list]

while True:
    ga[0].next_generation()
    best1 = ga[0].best_gene()
    if best1 is not None:
        print('gen1:{} best-value1:{}'.format(ga[0].generation_num, best1[1]))

    # ga2.next_generation()
    # best2 = ga2.best_gene()
    # if best2 is not None:
    #     print('gen2:{} best-value2:{}'.format(ga2.generation_num, best2[1]))

    # ga3.next_generation()
    # best3 = ga3.best_gene()
    # if best3 is not None:
    #     print('gen3:{} best-value3:{}'.format(ga3.generation_num, best3[1]))

    # ga4.next_generation()
    # best4 = ga4.best_gene()
    # if best4 is not None:
    #     print('gen4:{} best-value4:{}'.format(ga4.generation_num, best4[1]))

    # ga5.next_generation()
    # best5 = ga5.best_gene()
    # if best5 is not None:
    #     print('gen5:{} best-value5:{}'.format(ga5.generation_num, best5[1]))
