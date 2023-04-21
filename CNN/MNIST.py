import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from numpy import *

n_epochs = 3  #循环整个训练数据集的次数
batch_size_train = 64 #训练数据量的大小
batch_size_test = 1000
learning_rate = 0.01 #学习率
momentum = 0.5 #动量因子
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)#为cpu生成随机数的种子

start_epoch = -1#初始循环次数默认
mean = 0.1307 #归一化的均值
std = 0.3081 #方差

'''获取训练集'''
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([ #torchvision.transforms图像预处理；用Compose将多个步骤整合在一起，按顺序执行
                               torchvision.transforms.ToTensor(), #ToTensor转换，不管input的是PIL还是np, 只要是uint8格式的, 都会直接转成[0,1]
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))#Normalize用均值和标准差归一化张量图像
                             ])),
  batch_size=batch_size_train, shuffle=True)
#compose中：将图片转换为张量，图片进行归一化处理
'''Normalize: output = (input - mean) / std'''
'''获取测试集,标准差归一化处理'''
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

'''反归一化后的数据'''
examples = enumerate(test_loader) #enumerate() 函数用于将一个可遍历的数据对象
batch_idx, (example_data, example_targets) = next(examples) #example_targets 图片对应的实际标签
# print(example_targets)
# print(example_data[0][0])
# # print(example_data.shape)
# ######显示MNIST图像########################
# fig = plt.figure()
# for i in range(100):
#   plt.subplot(10,10,i+1) #创建2行3列单个子图
#   plt.tight_layout()#自动调整子图参数，使之填充整个图像区域
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   # plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()

target_ = []
pre_=[]

'''构建模型，初始化神经网络'''
class Net(nn.Module):
    def __init__(self):#构建三层神经网络
        super(Net, self).__init__()
        #实现2d卷积操作
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #nn.Conv2d 对由多个输入平面组成的输入信号进行二维卷积
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d() #随机将整个通道置0，默认元素置零概率0.5，防止过拟合
        self.fc1 = nn.Linear(320, 50)#用于设置网络中的全连接层 in_features输入的二维张量的大小
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):#前向传递，将数据传递到计算图中
        x = F.relu(F.max_pool2d(self.conv1(x), 2))#最大值池化，Relu激活
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)#view重构张量的维度，-1表示不确定几行，但确定320列；实现参数扁平化，便于全连接层输入
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)#正则化，对训练集的拟合有所损失
        x = self.fc2(x)
        return F.log_softmax(x)
        #Log_Softmax() 激活函数的值域是 ( − ∞ , 0 ]


'''初始化网络和优化器
SGD实现随机梯度下降算法,momentum动量 v = -dx*lr+momentum*v'''
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum) #parameters获取network的参数，网络的参数都保存在parameters()函数当中


'''模型训练'''
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

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
                #保存模型
            torch.save(network.state_dict(), './model.pth')#存放权重和偏置
            torch.save(optimizer.state_dict(), './optimizer.pth')

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

 test()
 for epoch in range(start_epoch + 2, n_epochs + 1):
  train(epoch)
  test()
 # '''============================模型加载及识别自己手写数字========================'''
 # continued_network = Net()
 # continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
 #                                 momentum=momentum)
 # '''加载模型'''
 # network_state_dict = torch.load('model.pth')
 # continued_network.load_state_dict(network_state_dict)
 # optimizer_state_dict = torch.load('optimizer.pth')
 # continued_optimizer.load_state_dict(optimizer_state_dict)
 #
 #  # 要识别的图片
 # input_image = 'D:/实验MyProjet/backward/img/9.bmp'
 # im = Image.open(input_image).resize((28, 28))  # 取图片数据
 # im = im.convert('L')  # 灰度图
 # im_data = np.array(im)
 #
 # im_data = torch.from_numpy(im_data).float()  # numpy->tensor
 # im_data = im_data.view(1, 1, 28, 28)
 # out = continued_network(im_data)
 # _, pred = torch.max(out, 1)
 # print('预测为:数字{}。'.format(pred))
 # print("-----------------------------------------错误图片--------------------------------------------")

 '''============寻找错误图片并对比归一化前真实图片============================='''

 for x, y in test_loader:
  output = network(x)
  pred = output.data.max(1, keepdim=True)[1]  # dim为1寻找每一行的最大值，keepdim保持维度不变 shape（1000,1）
  pre_ = np.array(pred).flatten()
  target_ = np.array(y.data).flatten()
  '''反归一化'''
  x_data = x
  for i in range(len(x)):
   x_data[i] = x[i] * std + mean
  # print(x_data[0][0])
  '''识别错误图片的位置'''
  error_ = []
  for i in range(len(target_)):
   if pre_[i] != target_[i]:
    error_.append(i)
  print(error_)
  print("len :", len(error_))
  '''显示错误图片'''
  for i in range(0, len(error_) + 1):
   if pre_[i] != target_[i]:
    print(i, ":", pre_[i], "...", target_[i])
    fig = plt.figure()
    plt.subplot(1, 2, 1)  # 创建2行3列单个子图
    plt.imshow(x[i][0], cmap='gray')
    # plt.imshow(example_data[i][0],cmap='gray')
    plt.title("False: {} -> Truth: {}".format(pre_[i], target_[i]))
    plt.subplot(1, 2, 2)
    plt.imshow(x_data[i][0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    # j+=1
    plt.show()








