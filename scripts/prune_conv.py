import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from channel_mask_generator import ChannelMaskGenerator
from dataset import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cloudpickle

# パラメータ利用
with open('CIFAR10_original_train.pkl', 'rb') as f:
    new_net = cloudpickle.load(f)

optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# 畳み込み層のリスト
conv_list = [new_net.features[i] for i in range(len(new_net.features)) if isinstance(new_net.features[i], nn.Conv2d)]

# 畳み込み層の数を計算
conv_count = len(conv_list)

# マスクのオブジェクト
with torch.no_grad():
    ch_mask = [ChannelMaskGenerator() for _ in range(conv_count)]

# 畳み込み層の出力数
conv_out = [conv.out_channels for conv in conv_list]

# 全結合パラメータの凍結
for param in new_net.classifier.parameters():
    param.requires_grad = False

# channel_pruning
count = 1
while count < 5:
    # ノルムの合計を保持
    pw_wlist = [list() for _ in range(conv_count)]

    # ノルムの取得
    with torch.no_grad():
        for i, conv in enumerate(conv_list):
            pw_wlist[i] = [np.sum(torch.abs(param).cpu().numpy()) for param in conv.weight]

    # 昇順にソート
    for i in range(len(pw_wlist)):
        pw_wlist[i].sort()

    # 刈る基準の閾値を格納
    pw_ratio = [pw_wlist[i][int(conv_out[i] / 5 * count)] for i in range(len(conv_out))]

    # 枝刈り本体
    with torch.no_grad():
        for i in range(len(conv_list)):
            conv_list[i].weight.data *= torch.tensor(ch_mask[i].generate_mask(conv_list[i].weight.data.clone(),
                                                                              None if i == 0 else conv_list[i - 1].weight.data.clone(),
                                                                              pw_ratio[i]), device=device, dtype=dtype)

        new_net.classifier[1].weight.data = torch.tensor(new_net.classifier[1].weight.data.clone().cpu().numpy()
                                                         * ch_mask[4].linear_mask(
            torch.t(new_net.classifier[1].weight.data.clone())), device=device, dtype=dtype)

    print()
    print(f'channel pruning: {count}')
    count += 1

    # パラメータの割合
    with torch.no_grad():
        weight_ratio = [np.count_nonzero(conv.weight.cpu().numpy()) / np.size(conv.weight.cpu().numpy()) for conv in conv_list]

    # 枝刈り後のチャネル数
    with torch.no_grad():
        channel_num_new = [conv_out[i] - ch_mask[i].channel_number(conv.weight) for i, conv in enumerate(conv_list)]

    for i in range(len(conv_list)):
        print(f'conv{i+1}_param: {weight_ratio[i]:.4f} ', end="")
    print()
    for i in range(len(conv_list)):
        print(f'channel_number{i+1}: {channel_num_new[i]} ', end="")
    print()

    f_num_epochs = 3
    # finetune
    for epoch in range(f_num_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        print("first", np.count_nonzero(new_net.features[0].weight.data.cpu().numpy()))
        print("first_mask", np.count_nonzero(ch_mask[0].mask))

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
            loss.backward()
            optimizer.step()
            if epoch == 0 and i == 0:
                print("train", np.count_nonzero(new_net.features[0].weight.data.cpu().numpy()))
                print("train_grad", np.count_nonzero(new_net.features[0].weight.grad.cpu().numpy()))

        avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)

        # val
        new_net.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                # view()での変換をしない
                images = images.to(device)
                labels = labels.to(device)
                outputs = new_net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)

        print(f'Epoch [{epoch + 1}/{f_num_epochs}], Loss: {avg_train_loss:.4f}, train_acc: {avg_train_acc:.4f}, '
              f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

    print("end", np.count_nonzero(new_net.features[0].weight.data.cpu().numpy()))
    print("end_mask", np.count_nonzero(ch_mask[0].mask))

# パラメータの保存
# with open('CIFAR10_conv_prune.pkl', 'wb') as f:
#     cloudpickle.dump(new_net, f)
