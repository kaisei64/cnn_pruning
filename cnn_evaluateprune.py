import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from channel_mask_generator import ChannelMaskGenerator
from dataset import *
import torch
import numpy as np
import cloudpickle

# 枝刈り前パラメータ利用
with open('./result/CIFAR10_original_train.pkl', 'rb') as f:
    original_net = cloudpickle.load(f)
# 畳み込み層のリスト
original_conv_list = [original_net.features[i] for i in range(len(original_net.features)) if
                      isinstance(original_net.features[i], nn.Conv2d)]


class CnnEvaluatePrune:
    def __init__(self):
        self.network = None

    def evaluate(self, gene, count, conv_num):
        return self.train(gene, count, conv_num)

    def train(self, gene, g_count, conv_num):
        # 枝刈り後パラメータ利用
        with open('./result/CIFAR10_conv_prune.pkl', 'rb') as f:
            self.network = cloudpickle.load(f)
        for param in self.network.classifier.parameters():
            param.requires_grad = False

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
            for i in range(len(conv_list)):
                if i == conv_num:
                    for j in range(len(conv_list[i].weight.data.cpu().numpy())):
                        if np.sum(np.abs(ch_mask[i].mask[j])) < 0.001:
                            ch_mask[i].mask[j] = 1
                            conv_list[i].weight.data[j] = torch.tensor(gene, device=device, dtype=dtype)
                            if i != len(conv_list) - 1:
                                ch_mask[i+1].mask[j, :] = 1
                                conv_list[i+1].weight.data[:, j] = original_conv_list[i+1].weight.data[:, j].clone()
                            add_count += 1
                            if add_count == 1:
                                print(f'add_filter_conv{conv_num + 1}')
                                break

        # パラメータの割合
        weight_ratio = [np.count_nonzero(conv.weight.cpu().detach().numpy()) /
                        np.size(conv.weight.cpu().detach().numpy()) for conv in conv_list]

        # 枝刈り後のチャネル数
        channel_num_new = [conv_list[i].out_channels - ch_mask[i].channel_number(conv.weight) for i, conv in
                           enumerate(conv_list)]

        print(f'parent{g_count + 1}: ') if g_count < 2 else print(f'children{g_count - 1}: ')
        for i in range(len(conv_list)):
            print(f'conv{i + 1}_param: {weight_ratio[i]:.4f}', end=", " if i != len(conv_list) - 1 else "\n")
        for i in range(len(conv_list)):
            print(f'channel_number{i + 1}: {channel_num_new[i]}', end=", " if i != len(conv_list) - 1 else "\n")

        f_num_epochs = 1
        eva = 0
        # finetune
        for epoch in range(f_num_epochs):
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

            print(f'epoch [{epoch + 1}/{f_num_epochs}], val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')
        return eva
