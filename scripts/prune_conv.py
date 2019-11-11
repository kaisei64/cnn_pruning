import os
import sys

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

from channel_mask_generator import ChannelMaskGenerator
from dense_mask_generator import DenseMaskGenerator

import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cloudpickle

device = 'cuda'
dtype = torch.float

# データの読み込み
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=2)

new_net = models.alexnet(num_classes=10)
optimizer1 = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# パラメータ利用
with open('CIFAR10_original_train.pkl', 'rb') as f:
    net = cloudpickle.load(f)


weight_ratio = []
for i in range(5):
    weight_ratio.append(100)

original_acc = 0
pruning_acc = 100
count = 1
pw_wlist = []
for i in range(5):
    pw_wlist.append([])

k = 1
p1 = ChannelMaskGenerator()
p2 = ChannelMaskGenerator()
p3 = ChannelMaskGenerator()
p4 = ChannelMaskGenerator()
p5 = ChannelMaskGenerator()
p6 = DenseMaskGenerator()
p7 = DenseMaskGenerator()

for i, param in enumerate(new_net.parameters()):
    if i >= 16:
        param.requires_grad = False

while weight_ratio[0] > 0.01 and count < 5 and pruning_acc > original_acc * 0.01:
    # channel_pruning
    for i in range(len(pw_wlist)):
        pw_wlist[i].clear()

    with torch.no_grad():
        cnt = 0
        for param in list(new_net.parameters()):
            if len(param.shape) == 4:
                for i in range(len(param)):
                    tmp = np.sum(torch.abs(param[i]).cpu().numpy())
                    pw_wlist[cnt].append(tmp)
                cnt += 1

    pw_idx = [new_net.conv1.out_channels / 20, new_net.conv2.out_channels / 20, new_net.conv3.out_channels / 20,
              new_net.conv4.out_channels / 20, new_net.conv5.out_channels / 20]

    pw_sort = []
    for i in range(len(pw_idx)):
        pw_sort.append([])
    for i in range(len(pw_idx)):
        pw_sort[i] = np.sort(pw_wlist[i], False)

    pw_ratio = []
    for i in range(len(pw_idx)):
        pw_ratio.append([])
    for i in range(len(pw_ratio)):
        if i == 0:
            pw_ratio[i] = pw_sort[i][int(pw_idx[i] * k)]
        elif i != 0:
            pw_ratio[i] = pw_sort[i][int(pw_idx[i] * k) - 1]

    k = k + 1

    with torch.no_grad():
        new_net.conv1.weight.data = p1.generate_mask(new_net.conv1.weight.data.clone(), None, pw_ratio[0])
        new_net.conv2.weight.data = p2.generate_mask(new_net.conv2.weight.data.clone(),
                                                     new_net.conv1.weight.data.clone(),
                                                     pw_ratio[1])
        new_net.conv3.weight.data = p3.generate_mask(new_net.conv3.weight.data.clone(),
                                                     new_net.conv2.weight.data.clone(),
                                                     pw_ratio[2])
        new_net.conv4.weight.data = p4.generate_mask(new_net.conv4.weight.data.clone(),
                                                     new_net.conv3.weight.data.clone(),
                                                     pw_ratio[3])
        new_net.conv5.weight.data = p5.generate_mask(new_net.conv5.weight.data.clone(),
                                                     new_net.conv4.weight.data.clone(),
                                                     pw_ratio[4])
        linear1_mask = torch.tensor(p5.linear_mask(torch.t(new_net.fc1.weight.data.clone())), device=device,
                                    dtype=dtype)
        new_net.fc1.weight.data = torch.tensor(new_net.fc1.weight.data.clone().cpu().numpy()
                                               * p5.linear_mask(torch.t(new_net.fc1.weight.data.clone())),
                                               device=device, dtype=dtype)
        linear2_mask = torch.tensor(np.ones(new_net.fc2.weight.data.clone().cpu().numpy().shape), device=device,
                                    dtype=dtype)
        p6.mask = torch.t(linear1_mask)
        p7.mask = torch.t(linear2_mask)

    print()
    print("first pruning:", count)
    count += 1
    cnt = 0

    weight_ratio[0] = np.count_nonzero(new_net.conv1.weight.data.cpu().numpy()) / np.size(
        new_net.conv1.weight.data.cpu().numpy())
    weight_ratio[1] = np.count_nonzero(new_net.conv2.weight.data.cpu().numpy()) / np.size(
        new_net.conv2.weight.data.cpu().numpy())
    weight_ratio[2] = np.count_nonzero(new_net.conv3.weight.data.cpu().numpy()) / np.size(
        new_net.conv3.weight.data.cpu().numpy())
    weight_ratio[3] = np.count_nonzero(new_net.conv4.weight.data.cpu().numpy()) / np.size(
        new_net.conv4.weight.data.cpu().numpy())
    weight_ratio[4] = np.count_nonzero(new_net.conv5.weight.data.cpu().numpy()) / np.size(
        new_net.conv5.weight.data.cpu().numpy())
    channel_number1 = new_net.conv1.out_channels - p1.channel_number(new_net.conv1.weight.data)
    channel_number2 = new_net.conv2.out_channels - p2.channel_number(new_net.conv2.weight.data)
    channel_number3 = new_net.conv3.out_channels - p3.channel_number(new_net.conv3.weight.data)
    channel_number4 = new_net.conv4.out_channels - p4.channel_number(new_net.conv4.weight.data)
    channel_number5 = new_net.conv5.out_channels - p5.channel_number(new_net.conv5.weight.data)
    print("weight_ratio_w1:", weight_ratio[0], " weight_ratio_w2:", weight_ratio[1], " weight_ratio_w3:",
          weight_ratio[2])
    print(" weight_ratio_w4", weight_ratio[3], " weight_ratio_w5:", weight_ratio[4])
    print("channel_number1:", channel_number1, "channel_number2:", channel_number2, "channel_number3:", channel_number3,
          "channel_number4:", channel_number4, "channel_number5:", channel_number5)

    f_num_epochs = 3
    # finetune
    for epoch in range(f_num_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        print("first", np.count_nonzero(new_net.conv1.weight.data.cpu().numpy()))
        print("first_mask", np.count_nonzero(p1.mask))

        # train
        new_net.train()
        for i, (images, labels) in enumerate(train_loader):
            # view()での変換をしない
            images, labels = images.to(device), labels.to(device)

            optimizer1.zero_grad()
            outputs = new_net(images, True, True, p1.get_mask(), p2.get_mask(), p3.get_mask(),
                              p4.get_mask(), p5.get_mask(), p6.get_mask(), p7.get_mask())
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer1.step()
            if epoch == 0 and i == 0:
                print("train", np.count_nonzero(new_net.conv1.weight.data.cpu().numpy()))
                print("train_grad", np.count_nonzero(new_net.conv1.weight.grad.cpu().numpy()))

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

        # val
        new_net.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                # view()での変換をしない
                images = images.to(device)
                labels = labels.to(device)
                outputs = new_net(images, True, True, p1.get_mask(), p2.get_mask(), p3.get_mask(),
                                  p4.get_mask(), p5.get_mask(), p6.get_mask(), p7.get_mask())
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_acc = val_acc / len(test_loader.dataset)
        pruning_acc = avg_val_acc

        print(f'Epoch [{epoch + 1}/{f_num_epochs}], Loss: {avg_train_loss:.4f}, train_acc: {avg_train_acc:.4f}, '
              f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

    print("end", np.count_nonzero(new_net.conv1.weight.data.cpu().numpy()))
    print("end_mask", np.count_nonzero(p1.mask))
