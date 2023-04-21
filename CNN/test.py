import torch
import torchvision.datasets as dset
import torchvision
from torch import optim
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from CNN.MNIST import Net
from torch.utils.data import ConcatDataset
from numpy import *
import numpy as np
batch_size_test = 10 #训练数据量的大小
batch_size_train = 2
learning_rate = 0.01 #学习率
momentum = 0.5 #动量因子
n_epochs = 3  #循环整个训练数据集的次数
log_interval = 10
mean = 0.1307 #归一化的均值
std = 0.3081 #方差
target_ = []
pre_=[]

mnist_loader = torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([ #torchvision.transforms图像预处理；用Compose将多个步骤整合在一起，按顺序执行
                               torchvision.transforms.ToTensor(), #ToTensor转换，不管input的是PIL还是np, 只要是uint8格式的, 都会直接转成[0,1]
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))#Normalize用均值和标准差归一化张量图像
                             ]))

transform=torchvision.transforms.Compose([ #torchvision.transforms图像预处理；用Compose将多个步骤整合在一起，按顺序执行
                               torchvision.transforms.Grayscale(1),#转化为单通道
                               torchvision.transforms.ToTensor(), #ToTensor转换，不管input的是PIL还是np, 只要是uint8格式的, 都会直接转成[0,1]
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))#Normalize用均值和标准差归一化张量图像
                             ])
dataset_train = dset.ImageFolder(root="D:/MyProjet/backward/img",transform=transform)
dataset_test = dset.ImageFolder(root="D:/MyProjet/backward/img_test",transform=transform)
# print(dataset_train.classes)#文件名
# print(dataset_train.imgs)#返回所有文件夹中的图片路径和类别
# print(dataset_train[0]
data_loader_train = torch.utils.data.DataLoader(dataset_train,batch_size = batch_size_train,shuffle=True)#训练集
data_loader_test = torch.utils.data.DataLoader(dataset_test,batch_size = batch_size_test,shuffle=True) #测试集

'''合并数据集'''
training_data = mnist_loader + dataset_train
train_loader = torch.utils.data.DataLoader(training_data,batch_size=32,shuffle=True)
# examples = enumerate(train_loader) #enumerate() 函数用于将一个可遍历的数据对象
# batch_idx, (example_data, example_targets) = next(examples) #example_targets 图片对应的实际标签
# print(shape(example_targets))

test_losses = []
train_losses = []
train_counter = []
test_counter = [i*len(data_loader_train.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    model.train()
    for batch_idx,(data, target) in enumerate(train_loader):
        # data_ = torch.cat((data, data1), dim=0)
        # target_ = torch.cat((target, target1), dim=0)  # 连接mnist与自定义数据集
        model_optimizer.zero_grad()  # 梯度置零，把loss关于weight的导数变为零
        output = model(data)  # 前向传播计算预测值
        loss = F.nll_loss(output, target)  # 计算误差并回传 ∑P(i)logQ(i)
        loss.backward()  # 反向传播计算梯度
        model_optimizer.step()  # 更新所有参数

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data) , len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))  # item返回张量元素的值
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset) ))


def test():
    test_loss = 0
    correct = 0
    tar = 0
    for data, target in data_loader_test:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # 计算误差并求和
        pred = output.data.max(1, keepdim=True)[1]  # dim为1寻找每一行的最大值，keepdim保持维度不变 shape（1000,1）
        correct += pred.eq(target.data.view_as(pred)).sum()  # 正确率

    test_loss /= len(data_loader_test.dataset)  # 总平均损失值
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader_test.dataset),
        100. * correct / len(data_loader_test.dataset)))

if __name__ == "__main__":
    model = Net()
    model_optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)
    model.load_state_dict(torch.load("model.pth"))
    model_optimizer.load_state_dict(torch.load("optimizer.pth"))
    model.eval()

    test()
    for epoch in range(1,n_epochs+1):
        train(epoch)
        test()
