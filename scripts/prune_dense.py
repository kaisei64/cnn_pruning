import os
import sys

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

from dense_mask_generator import DenseMaskGenerator
from dataset import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cloudpickle
# from draw_architecture import mydraw

device = 'cuda'
dtype = torch.float

# パラメータ利用
with open('CIFAR10_original_train.pkl', 'rb') as f:
    new_net = cloudpickle.load(f)

optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# 全結合層のリスト
dense_list = [new_net.classifier[i] for i in range(len(new_net.classifier)) if
              isinstance(new_net.classifier[i], nn.Linear)]

# 全結合層の数を計算
dense_count = len(dense_list)

# マスクのオブジェクト
with torch.no_grad():
    de_mask = [DenseMaskGenerator() for _ in dense_list]

# 全結合層の入出力数
dense_in = [dense.in_features for dense in dense_list]
dense_out = [dense.out_features for dense in dense_list]

# 全結合パラメータの凍結
for param in new_net.features.parameters():
    param.requires_grad = False

# weight_pruning
count = 1
while count < 20:
    # 全結合層を可視化
    # if count == 1 or count == 10 or count == 18:
    #     mydraw([torch.t(new_net.fc1.weight.data).cpu().numpy(), torch.t(new_net.fc2.weight.data).cpu().numpy()])

    # 重みを１次元ベクトル化
    with torch.no_grad():
        pw_wlist = [np.reshape(torch.abs(dense.weight.data.clone()).cpu().numpy(),
                               (1, dense_in[i] * dense_out[i])).squeeze() for i, dense in enumerate(dense_list)]

    # 昇順にソート
    for i in range(len(pw_wlist)):
        pw_wlist[i].sort()

    # 刈る基準の閾値を格納
    pw_ratio = [pw_wlist[i][int(dense_in[i] * dense_out[i] / 5 * count) - 1] for i in range(len(dense_out))]

    # 枝刈り本体
    save_mask = [list() for _ in range(len(dense_list))]
    with torch.no_grad():
        for i, dense in enumerate(dense_list):
            save_mask[i] = de_mask[i].generate_mask(dense.weight.data.clone(), pw_ratio[i])
            dense.weight.data *= torch.tensor(save_mask[i], device=device, dtype=dtype)

    print()
    print(f'weight pruning: {count}')
    count += 1

    # パラメータの割合
    with torch.no_grad():
        weight_ratio = [np.count_nonzero(dense.weight.cpu().numpy()) / np.size(dense.weight.cpu().numpy()) for dense in
                        dense_list]

    # 枝刈り後のニューロン数
    with torch.no_grad():
        neuron_num_new = [dense_in[i] - de_mask[i].neuron_number(torch.t(dense.weight)) for i, dense in enumerate(dense_list)]

    for i in range(len(dense_list)):
        print(f'dense{i+1}_param: {weight_ratio[i]:.4f} ', end="")
    print()
    for i in range(len(dense_list)):
        print(f'neuron_number{i+1}: {neuron_num_new[i]} ', end="")

    f_num_epochs = 3
    # finetune
    for epoch in range(f_num_epochs):
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

        # train
        new_net.train()
        for i, (images, labels) in enumerate(train_loader):
            # view()での変換をしない
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = new_net(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            # print(train_acc / ((i+1) * 64))
            loss.backward()
            optimizer.step()
            # 枝刈り本体
            with torch.no_grad():
                for j, dense in enumerate(dense_list):
                    dense.weight.data *= torch.tensor(save_mask[j], device=device, dtype=dtype)

        avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)

        # val
        new_net.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                # view()での変換をしない
                images, labels = images.to(device), labels.to(device)
                outputs = new_net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)

        print(f'Epoch [{epoch + 1}/{f_num_epochs}], Loss: {avg_train_loss:.4f}, train_acc: {avg_train_acc:.4f}, '
              f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

# パラメータの保存
# with open('CIFAR10_dense_prune.pkl', 'wb') as f:
#     cloudpickle.dump(new_net, f)
