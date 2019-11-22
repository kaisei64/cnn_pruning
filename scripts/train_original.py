import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from dataset import *
import torch
import torchvision.models as models
import torch.optim as optim
import pandas as pd
import cloudpickle
import time

net = models.alexnet(num_classes=100).to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
num_epochs = 150

data = {'epoch': [], 'time': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start = time.time()
for epoch in range(num_epochs):
    # train
    net.train()
    train_loss, train_acc = 0, 0
    for i, (images, labels) in enumerate(train_loader):
        # view()での変換をしない
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)

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
    data['epoch'].append(epoch + 1)
    data['time'].append(process_time)
    data['train_loss'].append(avg_train_loss)
    data['train_acc'].append(avg_train_acc)
    data['val_loss'].append(val_loss)
    data['val_acc'].append(avg_val_acc)
    df = pd.DataFrame.from_dict(data)
    df.to_csv('./result/result_cifar10.csv')

# パラメータの保存
with open('./result/CIFAR10_original_train.pkl', 'wb') as f:
    cloudpickle.dump(net, f)
