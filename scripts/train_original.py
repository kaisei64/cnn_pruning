import os
import sys

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

from dataset import *

import torch
import torchvision.models as models
import torch.optim as optim
import cloudpickle

net = models.alexnet(num_classes=10).to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
num_epochs = 15

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    # train
    net.train()
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
    with torch.no_grad():
        for images, labels in test_loader:
            # view()での変換をしない
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)
    original_acc = avg_val_acc

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, train_acc: {avg_train_acc:.4f}, '
          f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

# パラメータの保存
with open('CIFAR10_original_train.pkl', 'wb') as f:
    cloudpickle.dump(net, f)
