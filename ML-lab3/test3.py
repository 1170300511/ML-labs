import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from sklearn import datasets

from sklearn.datasets import load_iris

"""
mean1 = [1, 2]
mean2 = [7, 8]
cov1 = np.mat([[2, 1], [1, 2]])
cov2 = np.mat([[2, 1], [1, 1]])
data1 = np.random.multivariate_normal(mean1, cov1, 100)
data2 = np.random.multivariate_normal(mean2, cov1, 100)

f = open('test.csv', 'w')
for i in range(100):
    f.write(str(data1[i][0]) + "," + str(data1[i][1]) + "\n")
    f.write(str(data2[i][0]) + "," + str(data2[i][1]) + "\n")
f.close()
data = pd.read_csv('test.csv', header=None).values
x = []
y = []
for i in range(len(data)):
    x.append(data[i][0])
    y.append(data[i][1])
plt.scatter(x, y, marker='o', color='black')
plt.show()

a = np.mat([[]])

f = open("test.csv", 'r')
data = pd.read_csv("test.csv", header=None).values
# print(type(data[:, 1]))
# print(len(data[1, :]))
# print(data[1, :])
"""

"""
a = []
a.append([1, 0, 0])
a.append([2, 0, 0])
a = np.array(a)
print(len(a))

a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([1, 2, 3])
print(a * b)
"""

X = pd.read_csv('transactions10k.dat', header=6).values
X = X[:8000, :]
# 绘制数据分布图
ax = plt.subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

estimator = KMeans(n_clusters=4)  # 构造聚类器
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == 3]

print('cov1:' + str(np.cov(x0.T)))
print('cov2:' + str(np.cov(x1.T)))
print('cov3:' + str(np.cov(x2.T)))
print('cov4:' + str(np.cov(x3.T)))
print(
    'mean1:[' + str(sum(x0[:, 0]) / len(x0)) + ' ' + str(sum(x0[:, 1] / len(x0))) + str(sum(x0[:, 2] / len(x0))) + ']')
print(
    'mean2:[' + str(sum(x1[:, 0]) / len(x1)) + ' ' + str(sum(x1[:, 1] / len(x1))) + str(sum(x1[:, 2] / len(x0))) + ']')
print(
    'mean3:[' + str(sum(x2[:, 0]) / len(x2)) + ' ' + str(sum(x2[:, 1] / len(x2))) + str(sum(x2[:, 2] / len(x0))) + ']')
print(
    'mean4:[' + str(sum(x3[:, 0]) / len(x3)) + ' ' + str(sum(x3[:, 1] / len(x3))) + str(sum(x3[:, 2] / len(x0))) + ']')
plt.legend(loc=2)
ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], c="tomato", marker='o', label='label0')
ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], c="orange", marker='*', label='label1')
ax.scatter(x2[:, 0], x2[:, 1], x2[:, 2], c="tomato", marker='*', label='label2')
ax.scatter(x3[:, 0], x3[:, 1], x3[:, 2], c="lightgreen", marker='*', label='label3')
plt.legend(loc=2)
plt.show()
