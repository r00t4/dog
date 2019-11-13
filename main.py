import numpy as np
import cv2
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# import sys  
# sys.path.append(r"./drive/My\ drive/dl"); from dataset import Dataset
# sys.path.append(r"./drive/My\ drive/dl"); from model import NeuralNetwork

from dataset import Dataset
from model import NeuralNetwork

batch_size = 64
train_dataset = Dataset()
data_loader = DataLoader(train_dataset,
                         shuffle=True,
                         batch_size=batch_size)

net = torch.load('./model.pth')

net = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=0.001,
                      momentum=0.9)

nepoch = 5
# nepoch 10
print("nepoch: ", nepoch)

start_time = datetime.datetime.now()
print("start time: ", start_time)

for epoch in range(nepoch):
    running_loss = 0.0
    for X, Y in data_loader:
        optimizer.zero_grad()

        pred = net(X)

        loss = criterion(pred, Y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    running_loss /= len(data_loader)
    print(datetime.datetime.now(), epoch, running_loss)

    torch.save(net.state_dict(), './model.pth')

print("finished in: ", datetime.datetime.now() - start_time)