import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from channel_mask_generator import ChannelMaskGenerator
from dataset import *
from result_save_visualization import *
import torch
import torch.nn as nn
import numpy as np
import time

data_dict = {'epoch': [], 'time': [], 'val_loss': [], 'val_acc': []}

# パラメータ利用, 全結合パラメータの凍結
new_net = parameter_use('./result/CIFAR10_dense_prune.pkl')

# 畳み込み層のリスト
conv_list = [new_net.features[i] for i in range(len(new_net.features)) if isinstance(new_net.features[i], nn.Conv2d)]
conv_count = len(conv_list)
# マスクのオブジェクト
ch_mask = [ChannelMaskGenerator() for _ in range(conv_count)]
inv_prune_ratio = 50

# channel_pruning
for count in range(1, inv_prune_ratio):
    print(f'\nchannel pruning: {count}')
    # ノルムの合計を保持
    channel_l1norm_for_each_layer = [list() for _ in range(conv_count)]

    # ノルムの取得, 昇順にソート
    for i, conv in enumerate(conv_list):
        channel_l1norm_for_each_layer[i] = [np.sum(torch.abs(param).cpu().detach().numpy()) for param in conv.weight]
        channel_l1norm_for_each_layer[i].sort()

    # 枝刈り本体
    with torch.no_grad():
        for i in range(len(conv_list)):
            threshold = channel_l1norm_for_each_layer[i][int(conv_list[i].out_channels / inv_prune_ratio * count) - 1]
            save_mask = ch_mask[i].generate_mask(conv_list[i].weight.data.clone(),
                                                 None if i == 0 else conv_list[i - 1].weight.data.clone(), threshold)
            conv_list[i].weight.data *= torch.tensor(save_mask, device=device, dtype=dtype)

    # パラメータの割合
    weight_ratio = [np.count_nonzero(conv.weight.cpu().detach().numpy()) / np.size(conv.weight.cpu().detach().numpy())
                    for conv in conv_list]

    # 枝刈り後のチャネル数
    channel_num_new = [conv.out_channels - ch_mask[i].channel_number(conv.weight) for i, conv in enumerate(conv_list)]

    for i in range(conv_count):
        print(f'conv{i + 1}_param: {weight_ratio[i]:.4f}', end=", " if i != conv_count - 1 else "\n")
    for i in range(conv_count):
        print(f'channel_number{i + 1}: {channel_num_new[i]}', end=", " if i != conv_count - 1 else "\n")

    f_num_epochs = 1
    # finetune
    start = time.time()
    for epoch in range(f_num_epochs):
        # val
        new_net.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                labels = labels.to(device)
                outputs = new_net(images.to(device))
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)

        process_time = time.time() - start

        print(f'epoch [{epoch + 1}/{f_num_epochs}], time: {process_time:.4f}, '
              f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

        # 結果の保存
        input_data = [epoch + 1, process_time, avg_val_loss, avg_val_acc]
        result_save('./result/dense_conv_prune_not_prune_result.csv', data_dict, input_data)

# パラメータの保存
parameter_save('./result/CIFAR10_dense_conv_prune_not_train.pkl', new_net)
parameter_save('./result/CIFAR10_dense_conv_prune_not_train_copy.pkl', new_net)
