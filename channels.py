# import random
# import numpy as np
# from random import choice
# import cv2
# from matplotlib import pyplot as plt
#
# img = plt.imread('F:/test.jpg')
# print(img.shape)
#
# "224*224的矩阵m，数值为0-2"
# m = np.random.randint(0,3,(224,224))
# n = np.random.randint(0,255,(224,224))
# print(m)
# "0|255 224*224矩阵n"
# # n = np.zeros((224,224))
# # list = [0,255]
# # for i in range(224):
# #     for j in range(224):
# #         n[i][j]=choice(list)
# # print(n)
# for i in range(224):
#     for j in range(224):
#         img[i][j][m[i][j]]=n[i][j]
# plt.imshow(img)
# plt.show()
import random
import numpy as np
from random import choice
import cv2
from matplotlib import pyplot as plt

# img = plt.imread('/home/user/test.jpg')
# print(img.shape)
#
# "224*224的矩阵m，数值为0-2"
# m = np.random.randint(0,3,(224,224))
# print(m)
# "0|255 224*224矩阵n"
# n = np.zeros((224,224))
# list = [0,255]
# for i in range(224):
#     for j in range(224):
#         n[i][j]=choice(list)
# print(n)
# for i in range(224):
#     for j in range(224):
#         img[i][j][m[i][j]]=n[i][j]
# plt.imshow(img)
# plt.show()
import matplotlib.image as mpimg  # mpimg 用于读取图片
import time
x = mpimg.imread('/home/user/test.jpg')
shape = x.shape
chn = np.random.randint(0, 3, (shape[0], shape[1]))
value = np.random.randint(0, 255, (shape[0], shape[1]))
# 第一种写法
start1 = time.time()
xc = x.copy()
x_row_index = np.array([range(0, shape[0])]).reshape([shape[0], 1]).repeat(shape[1], axis=1).reshape(
    [shape[0] * shape[1]])
x_col_index = np.array([range(0, shape[1])]).reshape([1, shape[1]]).repeat(shape[0], axis=0).reshape(
    [shape[0] * shape[1]])
chn_index = chn.reshape([shape[0] * shape[1]])
x[x_row_index, x_col_index, chn_index] = value.reshape([shape[0] * shape[1]])
end1 = time.time()
print('running time:',str(end1-start1))
# 第二种写法
start2 = time.time()
for i in range(shape[0]):
    for j in range(shape[1]):
        xc[i][j][chn[i][j]] = value[i][j]
end2 = time.time()
print('running time:',str(end2-start2))
# 比较结果
diff = x - xc
print(np.min(diff), np.max(diff))