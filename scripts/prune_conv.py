import os
import sys

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

from channel_mask_generator import ChannelMaskGenerator
# from dense_mask_generator import DenseMaskGenerator

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cloudpickle

device = 'cuda'
dtype = torch.float

# データの前処理
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# データの読み込み
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=True,
                                             transform=transform_train,
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False,
                                            transform=transform_test,
                                            download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=2)

# パラメータ利用
with open('CIFAR10_original_train.pkl', 'rb') as f:
    new_net = cloudpickle.load(f)

optimizer1 = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

original_acc = 0
pruning_acc = 100
count = 1
k = 1
conv_count = 0
channel_num_new = []

# マスクのオブジェクト
ch_mask = []
with torch.no_grad():
    for i in range(len(new_net.features)):
        if isinstance(new_net.features[i], nn.Conv2d):
            ch_mask.append(ChannelMaskGenerator())

# 畳み込み層の数を計算
with torch.no_grad():
    for i in range(len(new_net.features)):
        if isinstance(new_net.features[i], nn.Conv2d):
            conv_count += 1

# パラメータの割合
weight_ratio = []
for i in range(conv_count):
    weight_ratio.append(100)

# ノルムの合計を保持
pw_wlist = []
for i in range(conv_count):
    pw_wlist.append([])

# 畳み込み層の入出力数
conv_in = []
conv_out = []
with torch.no_grad():
    for i in range(len(new_net.features)):
        if isinstance(new_net.features[i], nn.Conv2d):
            conv_in.append(new_net.features[i].in_channels)
    for i in range(len(new_net.features)):
        if isinstance(new_net.features[i], nn.Conv2d):
            conv_out.append(new_net.features[i].out_channels)

# 枝刈りの割合
pw_idx = []
for i, param in enumerate(conv_out):
    pw_idx.append(param / 20)

# 全結合パラメータの凍結
for i, param in enumerate(new_net.parameters()):
    if i >= 10:
        param.requires_grad = False

# channel_pruning
while weight_ratio[0] > 0.01 and count < 5 and pruning_acc > original_acc * 0.01:
    for i in range(len(pw_wlist)):
        pw_wlist[i].clear()

    # ノルムの取得
    with torch.no_grad():
        cnt = 0
        for param in new_net.features:
            if isinstance(param, nn.Conv2d):
                for i in range(len(param.weight)):
                    tmp = np.sum(torch.abs(param.weight[i]).cpu().numpy())
                    pw_wlist[cnt].append(tmp)
                cnt += 1

    # 昇順にソート
    pw_sort = []
    for i in range(len(pw_idx)):
        pw_sort.append([])
    for i in range(len(pw_idx)):
        pw_sort[i] = np.sort(pw_wlist[i], False)

    # 刈る基準の閾値を格納
    pw_ratio = []
    for i in range(len(pw_idx)):
        pw_ratio.append([])
    for i in range(len(pw_ratio)):
        if i == 0:
            pw_ratio[i] = pw_sort[i][int(pw_idx[i] * k)]
        elif i != 0:
            pw_ratio[i] = pw_sort[i][int(pw_idx[i] * k) - 1]

    k = k + 1

    # 枝刈り本体
    with torch.no_grad():
        new_net.features[0].weight.data *= torch.tensor(
            ch_mask[0].generate_mask(new_net.features[0].weight.data.clone(),
                                     None,
                                     pw_ratio[0]), device=device, dtype=dtype)
        new_net.features[3].weight.data *= torch.tensor(
            ch_mask[1].generate_mask(new_net.features[3].weight.data.clone(),
                                     new_net.features[0].weight.data.clone(),
                                     pw_ratio[1]), device=device, dtype=dtype)
        new_net.features[6].weight.data *= torch.tensor(
            ch_mask[2].generate_mask(new_net.features[6].weight.data.clone(),
                                     new_net.features[3].weight.data.clone(),
                                     pw_ratio[2]), device=device, dtype=dtype)
        new_net.features[8].weight.data *= torch.tensor(
            ch_mask[3].generate_mask(new_net.features[8].weight.data.clone(),
                                     new_net.features[6].weight.data.clone(),
                                     pw_ratio[3]), device=device, dtype=dtype)
        new_net.features[10].weight.data *= torch.tensor(
            ch_mask[4].generate_mask(new_net.features[10].weight.data.clone(),
                                     new_net.features[8].weight.data.clone(),
                                     pw_ratio[4]), device=device, dtype=dtype)
        # linear1_mask = torch.tensor(p5.linear_mask(torch.t(new_net.fc1.weight.data.clone())), device=device,
        #                             dtype=dtype)
        new_net.classifier[1].weight.data = torch.tensor(new_net.classifier[1].weight.data.clone().cpu().numpy()
                                                         * ch_mask[4].linear_mask(
            torch.t(new_net.classifier[1].weight.data.clone())), device=device, dtype=dtype)

    print()
    print("first pruning:", count)
    count += 1
    cnt = 0

    with torch.no_grad():
        cnt2 = 0
        for param in new_net.features:
            if isinstance(param, nn.Conv2d):
                weight_ratio[cnt] = np.count_nonzero(param.weight.cpu().numpy()) / np.size(param.weight.cpu().numpy())
                cnt2 += 1

    with torch.no_grad():
        cnt = 0
        for param in new_net.features:
            if isinstance(param, nn.Conv2d):
                channel_num_new.append(conv_out[cnt] - ch_mask[cnt].channel_number(param.weight))
                cnt += 1
    print("weight_ratio_w1:", weight_ratio[0], " weight_ratio_w2:", weight_ratio[1], " weight_ratio_w3:",
          weight_ratio[2])
    print(" weight_ratio_w4", weight_ratio[3], " weight_ratio_w5:", weight_ratio[4])
    print("channel_number1:", channel_num_new[0], "channel_number2:", channel_num_new[1], "channel_number3:",
          channel_num_new[2], "channel_number4:", channel_num_new[3], "channel_number5:", channel_num_new[4])

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

            optimizer1.zero_grad()
            outputs = new_net(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer1.step()
            if epoch == 0 and i == 0:
                print("train", np.count_nonzero(new_net.features[0].weight.data.cpu().numpy()))
                print("train_grad", np.count_nonzero(new_net.features[0].weight.grad.cpu().numpy()))

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

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
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_acc = val_acc / len(test_loader.dataset)
        pruning_acc = avg_val_acc

        print(f'Epoch [{epoch + 1}/{f_num_epochs}], Loss: {avg_train_loss:.4f}, train_acc: {avg_train_acc:.4f}, '
              f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

    print("end", np.count_nonzero(new_net.features[0].weight.data.cpu().numpy()))
    print("end_mask", np.count_nonzero(ch_mask[0].mask))
