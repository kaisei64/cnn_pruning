import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from channel_mask_generator import ChannelMaskGenerator
from dataset import *
import torch
import torch.optim as optim
import numpy as np


class Evaluate_prune():
    def __init__(self):
        self.network = None

    def evaluate(self, gene, count):
        return self.train(gene, count)

    def train(self, gene, g_count):
        self.network = AlexNet(num_classes).to(device)
        optimizer = optim.SGD(self.network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        eva = 0

        with torch.no_grad():
            # for i in range(len(list(self.network.parameters()))):
            #     list(self.network.parameters())[i] = list(new_net.parameters())[i]

            self.network.conv1.weight.data = new_net.conv1.weight.data.clone()
            self.network.conv2.weight.data = new_net.conv2.weight.data.clone()
            self.network.conv3.weight.data = new_net.conv3.weight.data.clone()
            self.network.conv4.weight.data = new_net.conv4.weight.data.clone()
            self.network.conv5.weight.data = new_net.conv5.weight.data.clone()
            self.network.conv1.bias.data = new_net.conv1.bias.data.clone()
            self.network.conv2.bias.data = new_net.conv2.bias.data.clone()
            self.network.conv3.bias.data = new_net.conv3.bias.data.clone()
            self.network.conv4.bias.data = new_net.conv4.bias.data.clone()
            self.network.conv5.bias.data = new_net.conv5.bias.data.clone()
            self.network.fc1.weight.data = new_net.fc1.weight.data.clone()
            self.network.fc2.weight.data = new_net.fc2.weight.data.clone()
            self.network.fc1.bias.data = new_net.fc1.bias.data.clone()
            self.network.fc2.bias.data = new_net.fc2.bias.data.clone()

        # p1 = Channel_Prune()
        # p2 = Channel_Prune()
        # p3 = Channel_Prune()
        # p4 = Channel_Prune()
        # p5 = Channel_Prune()
        # p6 = Pruning()
        # p7 = Pruning()

        p1.mask = np.ones(new_net.conv1.weight.data.shape)
        p1.mask = np.where(np.abs(new_net.conv1.weight.data.clone().cpu().numpy()) == 0, 0, 1)
        p2.mask = np.ones(new_net.conv2.weight.data.shape)
        p2.mask = np.where(np.abs(new_net.conv2.weight.data.clone().cpu().numpy()) == 0, 0, 1)
        p3.mask = np.ones(new_net.conv3.weight.data.shape)
        p3.mask = np.where(np.abs(new_net.conv3.weight.data.clone().cpu().numpy()) == 0, 0, 1)
        p4.mask = np.ones(new_net.conv4.weight.data.shape)
        p4.mask = np.where(np.abs(new_net.conv4.weight.data.clone().cpu().numpy()) == 0, 0, 1)
        p5.mask = np.ones(new_net.conv5.weight.data.shape)
        p5.mask = np.where(np.abs(new_net.conv5.weight.data.clone().cpu().numpy()) == 0, 0, 1)
        p6.mask = np.ones(new_net.fc1.weight.data.shape)
        p6.mask = np.where(np.abs(new_net.fc1.weight.data.clone().cpu().numpy()) == 0, 0, 1)
        p7.mask = np.ones(new_net.fc2.weight.data.shape)
        p7.mask = np.where(np.abs(new_net.fc2.weight.data.clone().cpu().numpy()) == 0, 0, 1)

        # if count == 1 or count == 10 or count == 18:
        #     mydraw([torch.t(self.network.fc1.weight.data).cpu().numpy(), torch.t(self.network.fc2.weight.data).cpu().numpy()])

        # add
        with torch.no_grad():
            add_count = 0
            if len(gene) == self.network.conv1.in_channels:
                for i, j in enumerate(self.network.conv1.weight.data.cpu().numpy()):
                    if np.sum(np.abs(p1.mask[i])) < 0.001:
                        p1.mask[i] = 1
                        p2.mask[:, i] = 1
                        self.network.conv1.weight.data[i] = torch.tensor(gene, device=device, dtype=dtype)
                        self.network.conv2.weight.data[:, i] = torch.tensor(
                            np.random.rand(self.network.conv2.out_channels,
                                           self.network.conv2.kernel_size[0],
                                           self.network.conv2.kernel_size[1]),
                            device=device, dtype=dtype)
                        add_count = add_count + 1
                        if add_count == 1:
                            print("add_filter_conv1")
                            break

            add_count = 0
            if len(gene) == self.network.conv2.in_channels:
                for i, j in enumerate(self.network.conv2.weight.data.cpu().numpy()):
                    if np.sum(np.abs(p2.mask[i])) < 0.001:
                        p2.mask[i] = 1
                        p3.mask[:, i] = 1
                        self.network.conv2.weight.data[i] = torch.tensor(gene, device=device, dtype=dtype)
                        self.network.conv3.weight.data[:, i] = torch.tensor(
                            np.random.rand(self.network.conv3.out_channels,
                                           self.network.conv3.kernel_size[0],
                                           self.network.conv3.kernel_size[1]),
                            device=device, dtype=dtype)
                        add_count = add_count + 1
                        if add_count == 1:
                            print("add_filter_conv2")
                            break

            add_count = 0
            if len(gene) == self.network.conv3.in_channels:
                for i, j in enumerate(self.network.conv3.weight.data.cpu().numpy()):
                    if np.sum(np.abs(p3.mask[i])) < 0.001:
                        p3.mask[i] = 1
                        p4.mask[:, i] = 1
                        self.network.conv3.weight.data[i] = torch.tensor(gene, device=device, dtype=dtype)
                        self.network.conv4.weight.data[:, i] = torch.tensor(
                            np.random.rand(self.network.conv4.out_channels,
                                           self.network.conv4.kernel_size[0],
                                           self.network.conv4.kernel_size[1]),
                            device=device, dtype=dtype)
                        add_count = add_count + 1
                        if add_count == 1:
                            print("add_filter_conv3")
                            break

            add_count = 0
            if len(gene) == self.network.conv4.in_channels:
                for i, j in enumerate(self.network.conv4.weight.data.cpu().numpy()):
                    if np.sum(np.abs(p4.mask[i])) < 0.001:
                        p4.mask[i] = 1
                        p5.mask[:, i] = 1
                        self.network.conv4.weight.data[i] = torch.tensor(gene, device=device, dtype=dtype)
                        self.network.conv5.weight.data[:, i] = torch.tensor(
                            np.random.rand(self.network.conv5.out_channels,
                                           self.network.conv5.kernel_size[0],
                                           self.network.conv5.kernel_size[1]),
                            device=device, dtype=dtype)
                        add_count = add_count + 1
                        if add_count == 1:
                            print("add_filter_conv4")
                            break

            # add_count = 0
            # for i, j in enumerate(self.network.conv5.weight.data.cpu().numpy()):
            #     print("i", i)
            #     print(np.sum(np.abs(p5.mask[i])))
            #     if np.sum(np.abs(p5.mask[i])) < 1:
            #         add_count = add_count+1
            #         print(p6.mask[:, i])
            # print(p6.mask[:, 0].shape)
            # print(add_count)
            add_count = 0
            if len(gene) == self.network.conv5.in_channels:
                for i, j in enumerate(self.network.conv5.weight.data.cpu().numpy()):
                    if np.sum(np.abs(p5.mask[i])) < 0.001:
                        p5.mask[i] = 1
                        p6.mask[:, i] = 1
                        self.network.conv5.weight.data[i] = torch.tensor(gene, device=device, dtype=dtype)
                        add_count = add_count + 1
                        if add_count == 1:
                            print("add_filter_conv5")
                            break

        ch1_num = self.network.conv1.out_channels - p1.channel_number(self.network.conv1.weight.data)
        ch2_num = self.network.conv2.out_channels - p2.channel_number(self.network.conv2.weight.data)
        ch3_num = self.network.conv3.out_channels - p3.channel_number(self.network.conv3.weight.data)
        ch4_num = self.network.conv4.out_channels - p4.channel_number(self.network.conv4.weight.data)
        ch5_num = self.network.conv5.out_channels - p5.channel_number(self.network.conv5.weight.data)
        weight_ratio[0] = np.count_nonzero(self.network.conv1.weight.data.cpu().numpy()) / np.size(
            new_net.conv1.weight.data.cpu().numpy())
        weight_ratio[1] = np.count_nonzero(self.network.conv2.weight.data.cpu().numpy()) / np.size(
            new_net.conv2.weight.data.cpu().numpy())
        weight_ratio[2] = np.count_nonzero(self.network.conv3.weight.data.cpu().numpy()) / np.size(
            new_net.conv3.weight.data.cpu().numpy())
        weight_ratio[3] = np.count_nonzero(self.network.conv4.weight.data.cpu().numpy()) / np.size(
            new_net.conv4.weight.data.cpu().numpy())
        weight_ratio[4] = np.count_nonzero(self.network.conv5.weight.data.cpu().numpy()) / np.size(
            new_net.conv5.weight.data.cpu().numpy())

        if g_count == 0:
            print("parent1:")
        elif g_count == 1:
            print("parent2:")
        elif g_count == 2:
            print("children1:")
        elif g_count == 3:
            print("children2:")
        print(" conv1:", ch1_num, " conv2:", ch2_num, " conv3:", ch3_num, "conv4:", ch4_num, "conv5:", ch5_num)
        print(" weight_ratio_w1:", weight_ratio[0], " weight_ratio_w2:", weight_ratio[1], " weight_ratio_w3:",
              weight_ratio[2])
        print(" weight_ratio_w4:", weight_ratio[3], " weight_ratio_w5:", weight_ratio[4])

        for i, param in enumerate(self.network.parameters()):
            if i <= 15:
                param.requires_grad = True
            if i >= 16:
                param.requires_grad = False

        f_num_epochs = 1
        # finetune
        for epoch in range(f_num_epochs):
            # train==============================
            self.network.train()
            train_loss, train_acc = 0, 0
            # ミニバッチで分割して読み込む
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                # 勾配をリセット
                optimizer.zero_grad()
                # 順伝播の計算
                outputs = self.network(images)
                # lossの計算
                loss = criterion(outputs, labels)
                # lossのミニバッチ分を溜め込む
                train_loss += loss.item()
                # accuracyをミニバッチ分を溜め込む
                train_acc += (outputs.max(1)[1] == labels).sum().item()
                # 逆伝播の計算
                loss.backward()
                # 重みの更新
                optimizer.step()
            # 平均lossと平均accuracyを計算
            avg_train_loss = train_loss / len(train_loader.dataset)
            avg_train_acc = train_acc / len(train_loader.dataset)

            # val==============================
            self.network.eval()
            val_loss, val_acc = 0, 0
            # 評価するときに必要のない計算が走らないようにtorch.no_gradを使用しています。
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.network(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += (outputs.max(1)[1] == labels).sum().item()
            avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)
            eva = avg_val_acc

            # 訓練データのlossと検証データのlossとaccuracyをログで出しています。
            print(f'epoch [{epoch + 1}/{f_num_epochs}], train_loss: {avg_train_loss:.4f}, train_acc:'
                  f' {avg_train_acc:.4f}, 'f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

        return eva
