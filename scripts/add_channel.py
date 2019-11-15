import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from pfga_cnn import PFGA
from evaluate_prune import Evaluate_prune
import cloudpickle

# パラメータ利用
with open('CIFAR10_original_train.pkl', 'rb') as f:
    new_net = cloudpickle.load(f)
ev = Evaluate_prune()
ga1 = PFGA(new_net.conv1.in_channels, new_net.conv1.kernel_size, evaluate_func=ev.evaluate, better_high=True,
           mutate_rate=0.1)
ga2 = PFGA(new_net.conv2.in_channels, new_net.conv2.kernel_size, evaluate_func=ev.evaluate, better_high=True,
           mutate_rate=0.1)
ga3 = PFGA(new_net.conv3.in_channels, new_net.conv3.kernel_size, evaluate_func=ev.evaluate, better_high=True,
           mutate_rate=0.1)
ga4 = PFGA(new_net.conv4.in_channels, new_net.conv4.kernel_size, evaluate_func=ev.evaluate, better_high=True,
           mutate_rate=0.1)
ga5 = PFGA(new_net.conv5.in_channels, new_net.conv5.kernel_size, evaluate_func=ev.evaluate, better_high=True,
           mutate_rate=0.1)

while True:
    ga1.next_generation()
    best1 = ga1.best_gene()
    if best1 is not None:
        print('gen1:{} best-value1:{}'.format(ga1.generation_num, best1[1]))

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
