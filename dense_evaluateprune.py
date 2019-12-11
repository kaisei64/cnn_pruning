from dense_mask_generator import DenseMaskGenerator
from result_save_visualization import *
from dataset import *
from neuron_importance import neuron_euclidean_distance, cos_sim
import torch
import numpy as np

data_dict = {'attribute': [], 'epoch': [], 'val_loss': [], 'val_acc': []}

# 枝刈り前パラメータ利用
original_net = parameter_use('./result/CIFAR10_original_train_epoch150.pkl')

# 全結合層のリスト
original_dense_list = [original_net.classifier[i] for i in range(len(original_net.classifier)) if
                       isinstance(original_net.classifier[i], nn.Linear)]


class DenseEvaluatePrune:
    def __init__(self):
        self.network = None

    def evaluate(self, gene, count, dense_num):
        return self.train(gene, count, dense_num)

    def train(self, gene, g_count, dense_num):
        # 枝刈り後パラメータ利用
        self.network = parameter_use('./result/CIFAR10_dense_conv_prune.pkl')
        for param in self.network.features.parameters():
            param.requires_grad = False

        # 全結合層のリスト
        dense_list = [self.network.classifier[i] for i in range(len(self.network.classifier))
                      if isinstance(self.network.classifier[i], nn.Linear)]

        # マスクのオブジェクト
        de_mask = [DenseMaskGenerator() for _ in range(len(dense_list))]
        for i, dense in enumerate(dense_list):
            de_mask[i].mask = np.where(np.abs(dense.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)

        # 追加
        with torch.no_grad():
            for j in range(len(dense_list[dense_num].weight.data.cpu().numpy())):
                if np.all(de_mask[dense_num].mask[:, j] == 0):
                    de_mask[dense_num].mask[:, j] = 1
                    dense_list[dense_num].weight.data[:, j] = torch.tensor(gene, device=device, dtype=dtype)
                    print(f'add_neuron_dense{dense_num + 1}')
                    break

        # パラメータの割合
        weight_ratio = [np.count_nonzero(dense.weight.cpu().detach().numpy()) /
                        np.size(dense.weight.cpu().detach().numpy()) for dense in dense_list]

        # 枝刈り後のニューロン数
        neuron_num_new = [dense_list[i].in_features - de_mask[i].neuron_number(dense.weight) for i, dense in
                          enumerate(dense_list)]

        print(f'parent{g_count + 1}: ') if g_count < 2 else print(f'children{g_count - 1}: ')
        for i in range(len(dense_list)):
            print(f'dense{i + 1}_param: {weight_ratio[i]:.4f}', end=", " if i != len(dense_list) - 1 else "\n")
        for i in range(len(dense_list)):
            print(f'neuron_number{i + 1}: {neuron_num_new[i]}', end=", " if i != len(dense_list) - 1 else "\n")

        similarity = 0
        # ニューロン間の類似度
        for i in range(dense_list[dense_num].in_features):
            similarity += neuron_euclidean_distance(gene, dense_list[dense_num].weight.data.cpu().detach().numpy()[i])
            # similarity += cos_sim(gene, dense_list[dense_num].weight.data.cpu().detach().numpy()[i])

        f_num_epochs = 1
        eva = 0
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
            eva = avg_val_loss

            print(f'epoch [{epoch + 1}/{f_num_epochs}], val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

            # 結果の保存
            input_data = [g_count, epoch + 1, avg_val_loss, avg_val_acc]
            result_save('./result/result_add_neurons_not_train.csv', data_dict, input_data)

        return eva + similarity
