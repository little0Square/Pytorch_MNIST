from itertools import chain
##############bp算法，实现平均误差图以及输出样本与期望值对比图
from numpy import *
import numpy as np
aver = []
outdata = []
class CBpnet:
    # 第一阶段 数据预处理阶段
    # 构造函数，输入训练数据trainx，输出数据y
    def __init__(self, trainx, trainy):
        self.hidenum = 2 #隐藏层节点的个数
        self.error = 1  #误差函数
        self.e = 0
        self.error_avg = 0 #平均误差
        self.learningrata = 0.075  # 默认学习率为0.9。学习率越小越好，但收敛速度越慢
        self.trainy = self.__normalize__(trainy)  # 训练输出数据归一化
        self.data1 = self.__normalize__(trainx)  # 训练输入数据归一化
        mx, nx = shape(trainx)

        # 1 用0作为初始化参数 测试
        self.weight1 = zeros((self.hidenum, mx))  # 默认隐藏层有self.hidenum个神经元，输入层与隐藏层的链接权值
        self.b1 = zeros((self.hidenum, 1))
        my, ny = shape(trainy)
        self.weight2 = zeros((my, self.hidenum))  # 隐藏层与输出层的链接权值
        self.b2 = zeros((my, 1))
        # 2、采用随机初始化为-1~1之间的数 测试
        self.weight1 = 2 * random.random((self.hidenum, mx)) - 1
        self.weight2 = 2 * random.random((my, self.hidenum)) - 1  # 隐藏层与输出层的链接权值

    # 训练数据归一化至[0,1]区间
    def __normalize__(self, trainx):  #快速归一化算法为线性转换算法
        minx, maxx = self.__MaxMin__(trainx)
        return (trainx - minx) / (maxx - minx)

    def __MaxMin__(self, trainX):
        n, m = shape(trainX)
        minx = zeros((n, 1))
        maxx = zeros((n, 1))
        for i in range(n):
            minx[i, 0] = trainX[i, :].min()
            maxx[i, 0] = trainX[i, :].max()
        return minx, maxx

    # 第二阶段 数据训练阶段
    def Traindata(self):
        mx, nx = shape(self.data1)
        # 随机梯度下降法
        for i in range(nx):
            # 第一步、前向传播
            # 隐藏层
            # print(self.data1[:, i])
            outdata2 = self.__ForwardPropagation__(self.data1[:, i], self.weight1, self.b1)  # 隐藏层节点数值
            # 输出层   outdatatemp为隐藏层数值
            outdata3 = self.__ForwardPropagation__(outdata2, self.weight2, self.b2)
            # print(outdata3)
            outdata3_ = list(np.array(outdata3).flatten())
            # print("\t",outdata3_)
            outdata.append(outdata3_)
            ####均方误差E(transpose即转置)
            # print(self.trainy[:, i])
            self.e = self.e + (outdata3 - self.trainy[:, i]).transpose() * (outdata3 - self.trainy[:, i])
            self.error = self.e / 2.
            self.error_avg = self.error_avg + self.error
            # print(i,"\t",self.error)
            # 计算每一层的误差值
            sigma3 = (1 - outdata3).transpose() * outdata3 * (outdata3 - self.trainy[:, i]) #δij=(1-f(x))*f(x)*(dj-yj)
            sigma2 = ((1 - outdata2).transpose() * outdata2)[0, 0] * (self.weight2.transpose() * sigma3) #δki=(1-f(x))*f(x)*wij*δij
            # 计算每一层的偏导数
            w_derivative2 = sigma3 * outdata2.transpose() #E 对wij的偏导数为δij*xi
            b_derivative2 = sigma3  #E对bj的偏导数为δij(即sigma3)

            w_derivative1 = sigma2 * self.data1[:, i].transpose() #E对wki的偏导数为δki*xk
            b_derivative1 = sigma2  #E对bj的偏导数为δki
            # 梯度下降公式
            self.weight2 = self.weight2 - self.learningrata * w_derivative2  # w = w - ∆w(i,j)=w-η*(∂E/∂wij)
            self.b2 = self.b2 - self.learningrata * b_derivative2  # b = b - ∆b(i,j)=b-η*(∂E/∂b)

            self.weight1 = self.weight1 - self.learningrata * w_derivative1
            self.b1 = self.b1 - self.learningrata * b_derivative1

        self.error_avg = self.error_avg / nx
        E_aver = list(np.array(self.error_avg).flatten())#将二维数组转化为一维数组
        # print(self.error_avg)
        # print("\t",E_aver)
        aver.append(E_aver)



   #前向传播函数
    def __ForwardPropagation__(self, indata, weight, b):
        outdata = weight * indata + b #输入
        outdata = 1. / (1 + exp(-outdata)) #outdata经过sigmoid函数输出
        return outdata


trainX=mat([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]).transpose()
trainY=mat([[0],[0.1429],[0.2857],[0.4286],[0.5714],[0.7143],[0.8571],[1.0000]]).transpose()

st=CBpnet(trainX,trainY)
i=0
while(i<1000):
    st.e=0
    st.Traindata()
    # print (i,"\t","\t","\t",st.error)
    i=i+1

aver_ = []
aver_ = list(np.array(aver).flatten()) #二维数组转变为一维数组
mxu = shape(aver_)
print(aver_)
print(shape(aver_))

import matplotlib.pyplot as plt


X = np.arange(0,1000) #x轴为迭代次数，步长为5
print(shape(X))
# Y = np.arange(0,0.5,0.05)
my_xticks = np.arange(0,1000,50)
my_xticks[0]=1 #横坐标从1开始
plt.xticks(my_xticks)
plt.xlabel("迭代次数")
plt.ylabel("平均误差")
plt.plot(X,aver_)
plt.legend()
plt.show()

# trainY_ = []
trainY_ = list(np.array(trainY).flatten())
print("期望值：",trainY_)
outdata_ = []
outdata_ = list(np.array(outdata).flatten())#样本输出值
# print(len(outdata_))
# print(outdata_)

print("样本值：",outdata_[-8:])
print()
trainY_x = []
for j in range(len(outdata_)):
    a = trainY_[j%8]
    trainY_x.append(a)
# print(trainY_x)
outdata_X = []
for i in range(len(outdata_)):
    x = i%8
    outdata_X.append(x)
# print(outdata_X)
# print(shape(outdata_))
x_outdata = np.arange(0,8)
y = np.arange(0,1,0.05)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#设置坐标轴
plt.xticks(x_outdata)
plt.yticks(y)
plt.plot(outdata_X,trainY_x,label='期望值')
plt.plot(x_outdata,outdata_[-8:],color='r',alpha=0.9,label='样本值')
plt.title('样本输出值与期望值对比图')
# plt.scatter(outdata_X,outdata_,color='orange')
plt.legend()
plt.show()