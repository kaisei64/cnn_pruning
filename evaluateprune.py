import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from channel_mask_generator import ChannelMaskGenerator
from dataset import *
import torch
import torch.optim as optim
import numpy as np
import cloudpickle

# パラメータ利用, 全結合パラメータの凍結
with open('CIFAR10_original_train.pkl', 'rb') as f:
    original_net = cloudpickle.load(f)


class EvaluatePrune:
    def __init__(self):
        self.network = None

    def evaluate(self, gene, count):
        return self.train(gene, count)

    def train(self, gene, g_count):
        # パラメータ利用
        with open('CIFAR10_conv_prune.pkl', 'rb') as f:
            self.network = cloudpickle.load(f)
        for param in self.network.classifier.parameters():
            param.requires_grad = False

        optimizer = optim.SGD(self.network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # 畳み込み層のリスト
        conv_list = [self.network.features[i] for i in range(len(self.network.features)) if
                     isinstance(self.network.features[i], nn.Conv2d)]

        # マスクのオブジェクト
        ch_mask = [ChannelMaskGenerator() for _ in range(len(conv_list))]
        for i, conv in enumerate(conv_list):
            ch_mask[i].mask = np.where(np.abs(conv.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)

        # add
        with torch.no_grad():
            add_count = 0
            if len(gene) == conv_list[0].in_channels:
                for i, j in enumerate(conv_list[0].weight.data.cpu().numpy()):
                    if np.sum(np.abs(ch_mask[0].mask[i])) < 0.001:
                        ch_mask[0].mask[i] = 1
                        ch_mask[1].mask[:, i] = 1
                        conv_list[0].weight.data[i] = torch.tensor(gene, device=device, dtype=dtype)
                        # conv_list[1].weight.data[:, i] = torch.tensor(
                        #     np.random.rand(conv_list[1].out_channels,
                        #                    conv_list[1].kernel_size[0],
                        #                    conv_list[1].kernel_size[1]),
                        #     device=device, dtype=dtype)
                        conv_list[1].weight.data[:, i] = original_net.features[3].weight.data[:, i].clone()
                        add_count = add_count + 1
                        if add_count == 1:
                            print("add_filter_conv1")
                            break

            add_count = 0
            if len(gene) == conv_list[1].in_channels:
                for i, j in enumerate(conv_list[1].weight.data.cpu().numpy()):
                    if np.sum(np.abs(ch_mask[1].mask[i])) < 0.001:
                        ch_mask[1].mask[i] = 1
                        ch_mask[2].mask[:, i] = 1
                        conv_list[1].weight.data[i] = torch.tensor(gene, device=device, dtype=dtype)
                        conv_list[2].weight.data[:, i] = torch.tensor(
                            np.random.rand(conv_list[2].out_channels,
                                           conv_list[2].kernel_size[0],
                                           conv_list[2].kernel_size[1]),
                            device=device, dtype=dtype)
                        add_count = add_count + 1
                        if add_count == 1:
                            print("add_filter_conv2")
                            break

            add_count = 0
            if len(gene) == conv_list[2].in_channels:
                for i, j in enumerate(conv_list[2].weight.data.cpu().numpy()):
                    if np.sum(np.abs(ch_mask[2].mask[i])) < 0.001:
                        ch_mask[2].mask[i] = 1
                        ch_mask[3].mask[:, i] = 1
                        conv_list[2].weight.data[i] = torch.tensor(gene, device=device, dtype=dtype)
                        conv_list[3].weight.data[:, i] = torch.tensor(
                            np.random.rand(conv_list[3].out_channels,
                                           conv_list[3].kernel_size[0],
                                           conv_list[3].kernel_size[1]),
                            device=device, dtype=dtype)
                        add_count = add_count + 1
                        if add_count == 1:
                            print("add_filter_conv3")
                            break

            add_count = 0
            if len(gene) == conv_list[3].in_channels:
                for i, j in enumerate(conv_list[3].weight.data.cpu().numpy()):
                    if np.sum(np.abs(ch_mask[3].mask[i])) < 0.001:
                        ch_mask[3].mask[i] = 1
                        ch_mask[4].mask[:, i] = 1
                        conv_list[3].weight.data[i] = torch.tensor(gene, device=device, dtype=dtype)
                        conv_list[4].weight.data[:, i] = torch.tensor(
                            np.random.rand(conv_list[4].out_channels,
                                           conv_list[4].kernel_size[0],
                                           conv_list[4].kernel_size[1]),
                            device=device, dtype=dtype)
                        add_count = add_count + 1
                        if add_count == 1:
                            print("add_filter_conv4")
                            break

            add_count = 0
            if len(gene) == conv_list[4].in_channels:
                for i, j in enumerate(conv_list[4].weight.data.cpu().numpy()):
                    if np.sum(np.abs(ch_mask[4].mask[i])) < 0.001:
                        ch_mask[4].mask[i] = 1
                        conv_list[4].weight.data[i] = torch.tensor(gene, device=device, dtype=dtype)
                        add_count = add_count + 1
                        if add_count == 1:
                            print("add_filter_conv5")
                            break

        # パラメータの割合
        weight_ratio = [
            np.count_nonzero(conv.weight.cpu().detach().numpy()) / np.size(conv.weight.cpu().detach().numpy())
            for conv in conv_list]

        # 枝刈り後のチャネル数
        channel_num_new = [conv_list[i].out_channels - ch_mask[i].channel_number(conv.weight) for i, conv in
                           enumerate(conv_list)]

        print(f'parent{g_count+1}: ')
        for i in range(len(conv_list)):
            print(f'conv{i + 1}_param: {weight_ratio[i]:.4f}', end=", " if i != len(conv_list) - 1 else "\n")
        for i in range(len(conv_list)):
            print(f'channel_number{i + 1}: {channel_num_new[i]}', end=", " if i != len(conv_list) - 1 else "\n")

        f_num_epochs = 1
        eva = 0
        # finetune
        for epoch in range(f_num_epochs):
            # train
            self.network.train()
            train_loss, train_acc = 0, 0
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.network(images)
                loss = criterion(outputs, labels)
                train_loss += loss.item()
                train_acc += (outputs.max(1)[1] == labels).sum().item()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for j, conv in enumerate(conv_list):
                        conv.weight.data *= torch.tensor(ch_mask[j].mask, device=device, dtype=dtype)
            avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)

            # val
            self.network.eval()
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    labels = labels.to(device)
                    outputs = self.network(images.to(device))
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += (outputs.max(1)[1] == labels).sum().item()
            avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)
            eva = avg_val_acc

            print(
                f'epoch [{epoch + 1}/{f_num_epochs}], train_loss: {avg_train_loss:.4f}, train_acc: {avg_train_acc:.4f}'
                f', val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')
        return eva
