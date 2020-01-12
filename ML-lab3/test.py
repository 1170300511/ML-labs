import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib import pyplot as plt
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.datasets import load_iris

"""
mean1 = [1, 5]
mean2 = [5, 5]
mean3 = [5, 1]
mean4 = [1, 1]
cov1 = np.mat([[1, 0], [0, 1]])
data1 = np.random.multivariate_normal(mean1, cov1, 50)
data2 = np.random.multivariate_normal(mean2, cov1, 50)
data3 = np.random.multivariate_normal(mean3, cov1, 50)
data4 = np.random.multivariate_normal(mean4, cov1, 50)

f = open('test.csv', 'w')
for i in range(50):
    f.write(str(data1[i][0]) + "," + str(data1[i][1]) + "\n")
    f.write(str(data2[i][0]) + "," + str(data2[i][1]) + "\n")
    f.write(str(data3[i][0]) + "," + str(data3[i][1]) + "\n")
    f.write(str(data4[i][0]) + "," + str(data4[i][1]) + "\n")
f.close()
plt.legend(loc=2)
plt.scatter(data1[:, 0], data1[:, 1], marker='o', c='lightskyblue', label='class1')
plt.scatter(data2[:, 0], data2[:, 1], marker='*', c='tomato', label='class2')
plt.scatter(data3[:, 0], data3[:, 1], marker='o', c='lightgreen', label='class3')
plt.scatter(data4[:, 0], data4[:, 1], marker='*', c='orange', label='class4')
plt.scatter(1, 1, marker='+', c='blue', label='center', s=100)
plt.scatter(1, 5, marker='+', c='blue', s=100)
plt.scatter(5, 1, marker='+', c='blue', s=100)
plt.scatter(5, 5, marker='+', c='blue', s=100)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

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
"""
# 鸢尾花
iris = load_iris()

X = iris.data[:]  # 表示我们只取特征空间中的后两个维度

# 绘制数据分布图

plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')

plt.xlabel('petal length')

plt.ylabel('petal width')

plt.legend(loc=2)

plt.show()

estimator = KMeans(n_clusters=3)  # 构造聚类器

estimator.fit(X)  # 聚类

label_pred = estimator.labels_  # 获取聚类标签

# 绘制k-means结果

x0 = X[label_pred == 0]

x1 = X[label_pred == 1]

x2 = X[label_pred == 2]

plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')

plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')

plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')

plt.xlabel('petal length')

plt.ylabel('petal width')

plt.legend(loc=2)

plt.show()

"""

data = pd.read_csv('transactions10k.dat', header=6).values
data = data[:800, :]
# 创建一个绘图工程
ax = plt.subplot(111, projection='3d')
plt.legend(loc=2)
# 将数据点分成三部分画，在颜色上有区分度
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='green', label='class1')  # 绘制数据点
ax.set_zlabel('z')  # 坐标轴
ax.set_ylabel('y')  # 坐标轴
ax.set_xlabel('x')
plt.legend(loc=2)
plt.show()
