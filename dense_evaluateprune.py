import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from dense_mask_generator import DenseMaskGenerator
from dataset import *
import torch
import numpy as np
import cloudpickle

# 枝刈り前パラメータ利用
with open('./result/CIFAR10_original_train.pkl', 'rb') as f:
    original_net = cloudpickle.load(f)
# 畳み込み層のリスト
original_dense_list = [original_net.classifier[i] for i in range(len(original_net.classifier)) if
                       isinstance(original_net.classifier[i], nn.Linear)]


class DenseEvaluatePrune:
    def __init__(self):
        self.network = None

    def evaluate(self, gene, count, dense_num):
        return self.train(gene, count, dense_num)

    def train(self, gene, g_count, dense_num):
        # 枝刈り後パラメータ利用
        with open('./result/CIFAR10_dense_prune.pkl', 'rb') as f:
            self.network = cloudpickle.load(f)
        for param in self.network.features.parameters():
            param.requires_grad = False

        # 畳み込み層のリスト
        dense_list = [self.network.classifier[i] for i in range(len(self.network.classifier))
                      if isinstance(self.network.classifier[i], nn.Linear)]

        # マスクのオブジェクト
        de_mask = [DenseMaskGenerator() for _ in range(len(dense_list))]
        for i, dense in enumerate(dense_list):
            de_mask[i].mask = np.where(np.abs(dense.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)

        # add
        # with torch.no_grad():
        #     add_count = 0
        #     for i in range(len(conv_list)):
        #         if i == conv_num:
        #             for j in range(len(conv_list[i].weight.data.cpu().numpy())):
        #                 if np.sum(np.abs(ch_mask[i].mask[j])) < 0.001:
        #                     ch_mask[i].mask[j] = 1
        #                     conv_list[i].weight.data[j] = torch.tensor(gene, device=device, dtype=dtype)
        #                     if i != len(conv_list) - 1:
        #                         ch_mask[i+1].mask[j, :] = 1
        #                         conv_list[i+1].weight.data[:, j] = original_conv_list[i+1].weight.data[:, j].clone()
        #                     add_count += 1
        #                     if add_count == 1:
        #                         print(f'add_filter_conv{conv_num + 1}')
        #                         break

        # パラメータの割合
        weight_ratio = [np.count_nonzero(dense.weight.cpu().detach().numpy()) /
                        np.size(dense.weight.cpu().detach().numpy()) for dense in dense_list]

        # 枝刈り後のチャネル数
        neuron_num_new = [dense_list[i].out_channels - de_mask[i].neuron_number(dense.weight) for i, dense in
                          enumerate(dense_list)]

        print(f'parent{g_count + 1}: ') if g_count < 2 else print(f'children{g_count - 1}: ')
        for i in range(len(dense_list)):
            print(f'dense{i + 1}_param: {weight_ratio[i]:.4f}', end=", " if i != len(dense_list) - 1 else "\n")
        for i in range(len(dense_list)):
            print(f'neuron_number{i + 1}: {neuron_num_new[i]}', end=", " if i != len(dense_list) - 1 else "\n")

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
