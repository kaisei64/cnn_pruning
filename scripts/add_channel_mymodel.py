import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from channel_mask_generator import ChannelMaskGenerator
from dense_mask_generator import DenseMaskGenerator
from dataset import *
from pfgacnn_mymodel import PfgaCnn
from cnn_evaluateprune_mymodel import CnnEvaluatePrune
from result_save_visualization import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

data_dict = {'before_val_loss': [], 'before_val_acc': [], 'val_loss': [], 'val_acc': []}
# 枝刈り前パラメータ利用
original_net = parameter_use('./result/original_train_epoch150_mymodel.pkl')
# 枝刈り前畳み込み層のリスト
original_conv_list = [module for module in original_net.modules() if isinstance(module, nn.Conv2d)]
# 枝刈り後パラメータ利用
new_net = parameter_use('./result/dense_conv_prune_mymodel_dense90per_20per.pkl')
# 枝刈り後畳み込み層・全結合層・係数パラメータのリスト
conv_list = [module for module in new_net.modules() if isinstance(module, nn.Conv2d)]
dense_list = [module for module in new_net.modules() if isinstance(module, nn.Linear)]
param_list = [module for module in new_net.modules() if isinstance(module, nn.Parameter)]
# マスクのオブジェクト
ch_mask = [ChannelMaskGenerator() for _ in range(len(conv_list))]
for i, conv in enumerate(conv_list):
    ch_mask[i].mask = np.where(np.abs(conv.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)
de_mask = [DenseMaskGenerator() for _ in range(len(dense_list))]
for i, dense in enumerate(dense_list):
    de_mask[i].mask = np.where(np.abs(dense.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)

gen_num = 30
add_channel_num = 3
optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# 追加前重み分布の描画
for i in range(len(conv_list)):
    before_weight = [np.sum(conv_list[i].weight.data[k].cpu().detach().numpy()) for k
                     in range(len(conv_list[i].weight.data.cpu().numpy()))]
    parameter_distribution_vis(f'./figure/dis_vis_mymodel/conv{i + 1}/before_weight_distribution{i + 1}.png',
                               before_weight)

for count in range(add_channel_num):
    ev = [CnnEvaluatePrune(count) for _ in range(len(conv_list))]
    ga = [PfgaCnn(conv.in_channels, conv.kernel_size, i,
                  evaluate_func=ev[i].evaluate, better_high=False, mutate_rate=0.1) for i, conv in enumerate(conv_list)]
    best = [list() for _ in range(len(ga))]
    for i in range(len(ga)):
        # if i == 0 and count % 6 != 0 or i == 1 and count % 2 != 0 or i == 3 and count % 4 != 0 or i == 4 and count % 4 != 0:
        #     continue
        while ga[i].generation_num < gen_num:
            ga[i].next_generation()
            best[i] = ga[i].best_gene()
            if best[i] is not None:
                print(f'gen{i + 1}:{ga[i].generation_num} best-value{i + 1}:{best[i][1]}\n')

        with torch.no_grad():
            # 層ごとに１チャネルごと追加
            for j in range(len(conv_list[i].weight.data.cpu().numpy())):
                if np.sum(np.abs(ch_mask[i].mask[j])) < 25 * (count + 1) + 1:
                    ch_mask[i].mask[j] = 1
                    conv_list[i].weight.data[j] = torch.tensor(best[i][0], device=device, dtype=dtype)
                    if i != len(conv_list) - 1:
                        ch_mask[i + 1].mask[j, :] = 1
                        conv_list[i + 1].weight.data[:, j] = original_conv_list[i + 1].weight.data[:, j].clone()
                    break

            # 追加後重み分布の描画
            after_weight = [np.sum(conv_list[i].weight.data[k].cpu().numpy()) for k
                            in range(len(conv_list[i].weight.data.cpu().numpy()))]
            parameter_distribution_vis(
                f'./figure/dis_vis_mymodel/conv{i + 1}/after{count + 1}_weight_distribution{i + 1}.png', after_weight)

            # 追加後チャネル可視化
            # for j in range(conv_list[i].out_channels):
            #     conv_vis(f'./figure/ch_vis_mymodel/conv{i + 1}/after{count + 1}_conv{i + 1}_filter{j + 1}.png'
            #              , conv_list[i].weight.data.cpu().numpy(), j)

        # パラメータの保存
        parameter_save('./result/dense_conv_prune_mymodel_dense90per_20per.pkl', new_net)

    for param in new_net.parameters():
        param.requires_grad = False
    for dense in dense_list:
        dense.weight.requires_grad = True
    for param in param_list:
        param.requires_grad = True
    f_num_epochs = 1
    before_avg_val_loss, before_avg_val_acc = 0, 0
    # finetune
    for epoch in range(f_num_epochs):
        # val
        new_net.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                labels = labels.to(device)
                outputs = new_net(images.to(device), True)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        before_avg_val_loss, before_avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)

    for epoch in range(f_num_epochs):
        # train
        new_net.train()
        train_loss, train_acc = 0, 0
        for _, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = new_net(images, True)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()
            for param in param_list:
                param_max, param_min = torch.max(param), torch.min(param)
                param = 2 * (param - param_min) / (param_max - param_min)
                # print(param)
                # print(torch.max(param))
                # print(torch.min(param))
            with torch.no_grad():
                for j, dense in enumerate(dense_list):
                    if de_mask[j].mask is None:
                        break
                    dense.weight.data *= torch.tensor(de_mask[j].mask, device=device, dtype=dtype)
        avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)

        # val
        new_net.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                labels = labels.to(device)
                outputs = new_net(images.to(device), True)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)

        print(f'finetune, epoch [{epoch + 1}/{f_num_epochs}], train_loss: {avg_train_loss:.4f}'
              f', train_acc: {avg_train_acc:.4f}, val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')
        print()

        # 結果の保存
        input_data = [before_avg_val_loss, before_avg_val_acc, avg_val_loss, avg_val_acc]
        result_save('./result/result_add_channels_retrain.csv', data_dict, input_data)
        # パラメータの保存
        parameter_save('./result/dense_conv_prune_mymodel_dense90per_20per.pkl', new_net)
