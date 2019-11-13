import os
import sys

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

from dense_mask_generator import DenseMaskGenerator

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cloudpickle
# from draw_architecture import mydraw

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

optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

original_acc = 0
pruning_acc = 100
count = 1
k = 1
dense_count = 0

# マスクのオブジェクト
de_mask = []
with torch.no_grad():
    for i in range(len(new_net.classifier)):
        if isinstance(new_net.classifier[i], nn.Linear):
            de_mask.append(DenseMaskGenerator())

# 全結合層の数を計算
with torch.no_grad():
    for i in range(len(new_net.classifier)):
        if isinstance(new_net.classifier[i], nn.Linear):
            dense_count += 1

# パラメータの割合
weight_ratio = []
for i in range(dense_count):
    weight_ratio.append(100)

# 枝刈り後のニューロン数
neuron_num_new = []
for i in range(dense_count):
    neuron_num_new.append(0)

# 重みの１次元ベクトルを保持
pw_wlist = []
for i in range(dense_count):
    pw_wlist.append([])

# 全結合層の入出力数
dense_in = []
dense_out = []
with torch.no_grad():
    for i in range(len(new_net.classifier)):
        if isinstance(new_net.classifier[i], nn.Linear):
            dense_in.append(new_net.classifier[i].in_features)
    for i in range(len(new_net.classifier)):
        if isinstance(new_net.classifier[i], nn.Linear):
            dense_out.append(new_net.classifier[i].out_features)

# 枝刈りの割合
pw_idx = []
for param_in, param_out in zip(dense_in, dense_out):
    pw_idx.append(param_in * param_out / 20)

# 全結合パラメータの凍結
for param in new_net.features.parameters():
    param.requires_grad = False

# weight_pruning
while weight_ratio[0] > 0.01 and count < 20 and pruning_acc > original_acc * 0.05:
    # if count == 1 or count == 10 or count == 18:
    #     mydraw([torch.t(new_net.fc1.weight.data).cpu().numpy(), torch.t(new_net.fc2.weight.data).cpu().numpy()])
    for i in range(len(pw_wlist)):
        pw_wlist[i] = []

    # 重みを１次元ベクトル化
    with torch.no_grad():
        cnt = 0
        for param in new_net.classifier:
            if isinstance(param, nn.Linear):
                pw_wlist[cnt] = np.reshape(torch.abs(param.weight.data.clone()).cpu().numpy(),
                                           (1, dense_in[cnt] * dense_out[cnt])).squeeze()
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
        pw_ratio[i] = pw_sort[i][int(pw_idx[i] * k) - 1]
    k = k + 1

    # 枝刈り本体
    with torch.no_grad():
        cnt = 0
        for param in new_net.classifier:
            if isinstance(param, nn.Linear):
                param.weight *= torch.tensor(de_mask[cnt].generate_mask(param.weight.data.clone(), pw_ratio[cnt])
                                             , device=device, dtype=dtype)
                cnt += 1

    print()
    print(f'weight pruning: {count}')
    count += 1

    with torch.no_grad():
        cnt = 0
        for param in new_net.classifier:
            if isinstance(param, nn.Linear):
                weight_ratio[cnt] = np.count_nonzero(param.weight.cpu().numpy()) / np.size(param.weight.cpu().numpy())
                cnt += 1

    with torch.no_grad():
        cnt = 0
        for param in new_net.classifier:
            if isinstance(param, nn.Linear):
                neuron_num_new[cnt] = dense_out[cnt] - de_mask[cnt].neuron_number(param.weight)
                cnt += 1
    print(f'dense1_param: {weight_ratio[0]:.4f}, dense2_param: {weight_ratio[1]:.4f}'
          f', dense3_param: {weight_ratio[2]:.4f}')
    print(f'neuron_number1: {neuron_num_new[0]}, neuron_number2: {neuron_num_new[1]}, neuron_number3: '
          f'{neuron_num_new[2]}')

    f_num_epochs = 10
    # finetune
    for epoch in range(f_num_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

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

# パラメータの保存
# with open('CIFAR10_dense_prune.pkl', 'wb') as f:
#     cloudpickle.dump(new_net, f)
