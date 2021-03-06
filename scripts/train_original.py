import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from dataset import *
from result_save_visualization import *
import torch
import torchvision.models as models
import torch.optim as optim
import time

net = models.alexnet(num_classes=10).to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
num_epochs = 150

data_dict = {'epoch': [], 'time': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start = time.time()
for epoch in range(num_epochs):
    # train
    net.train()
    train_loss, train_acc = 0, 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()
    avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)

    # val
    net.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            # view()での変換をしない
            labels = labels.to(device)
            outputs = net(images.to(device))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)
    process_time = time.time() - start

    print(f'epoch [{epoch + 1}/{num_epochs}], time: {process_time:.4f}, train_loss: {avg_train_loss:.4f}'
          f', train_acc: {avg_train_acc:.4f}, 'f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

    # 結果の保存
    input_data = [epoch + 1, process_time, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc]
    result_save('./result/result_cifar10_epoch150.csv', data_dict, input_data)

# パラメータの保存
parameter_save('./result/CIFAR10_original_train_epoch150.pkl', net)

# 勾配の保存
conv_list = [net.features[i] for i in range(len(net.features)) if
             isinstance(net.features[i], nn.Conv2d)]
dense_list = [net.classifier[i] for i in range(len(net.classifier)) if
              isinstance(net.classifier[i], nn.Linear)]
for i, conv in enumerate(conv_list):
    parameter_save(f'./result/original_grad_conv{i}.pkl', conv.weight.grad)
for i, dense in enumerate(dense_list):
    parameter_save(f'./result/original_grad_dense{i}.pkl', dense.weight.grad)
