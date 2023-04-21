import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from CNN.MNIST import Net

log_interval = 10
batch_size_train = 64 #训练数据量的大小
learning_rate = 0.01 #学习率
momentum = 0.5 #动量因子

'''获取训练集'''
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([ #torchvision.transforms图像预处理；用Compose将多个步骤整合在一起，按顺序执行
                               torchvision.transforms.ToTensor(), #ToTensor转换，不管input的是PIL还是np, 只要是uint8格式的, 都会直接转成[0,1]
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))#Normalize用均值和标准差归一化张量图像
                             ])),
  batch_size=batch_size_train, shuffle=True)

train_losses = []
train_counter = []

'''定义模型函数'''
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad() #梯度置零，把loss关于weight的导数变为零
        output = network(data) #前向传播计算预测值
        loss = F.nll_loss(output, target) #计算误差并回传 ∑P(i)logQ(i)
        loss.backward() #反向传播计算梯度
        optimizer.step()#更新所有参数

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item())) #item返回张量元素的值
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            #     #保存模型
            # torch.save(network.state_dict(), './model.pth')#存放权重和偏置
            # torch.save(optimizer.state_dict(), './optimizer.pth')

if __name__ == "__main__":
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
    network.load_state_dict(torch.load("model.pth"))
    optimizer.load_state_dict(torch.load("optimizer.pth"))
    network.eval()

    train(epoch=1)
