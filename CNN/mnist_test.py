import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from CNN.MNIST import Net
from CNN.MNIST import train_loader

n_epochs = 3  #循环整个训练数据集的次数
batch_size_test = 1000
learning_rate = 0.01 #学习率
momentum = 0.5 #动量因子

'''获取测试集,标准差归一化处理'''
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

'''定义测试函数'''
def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():#不要求计算梯度，不会进行行反向传播
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() #计算误差并求和
            pred = output.data.max(1, keepdim=True)[1] #dim为1寻找每一行的最大值，keepdim保持维度不变 shape（1000,1）
            correct += pred.eq(target.data.view_as(pred)).sum()#正确率

    test_loss /= len(test_loader.dataset) #总平均损失值
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
    network.load_state_dict(torch.load("model.pth"))
    optimizer.load_state_dict(torch.load("optimizer.pth"))
    network.eval()

    test()