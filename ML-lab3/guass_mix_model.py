import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

MAX_INT = 10000000  # 初始化为一个较大的值
DOOR = 1e-30  # k-means算法迭代结束的阈值


class GaussMix:
    """高斯混合模型"""

    def __init__(self):
        """k-means的参数"""

        # self.data = pd.read_csv("test.csv", header=None).values  # 读入多样本点
        self.data = pd.read_csv('transactions10k.dat', header=6).values
        self.data = self.data[:8000, :]  # 由于数据量太大，所以只选取其中的一部分作为研究
        self.K = 4  # 聚几类
        self.dim = len(self.data[0, :])  # 数据的维度
        # self.center = np.tile(np.array([[.0, .0]]), (self.K, 1))  # 存储中心点,使用np.array初始化数组时要注意：要定义为浮点数，否则会直接取整
        self.center = np.tile(np.array([[.0, .0, .0]]), (self.K, 1))  # 用于测试集
        self.n = len(self.data)  # 样本数量
        self.flags = list(range(self.n))  # 记录每一个样本点属于哪一族
        self.flags_em = list(range(self.n))

        """高斯混合模型新增的参数"""

        # self.cov_all = [np.array([[2., 1.], [1., 2.]]), np.array([[3., 1.], [1., 4.]]),
        # np.array([[2., 1.], [1., 2.]]),
        # np.array([[3., 1.], [1., 4.]])]  # 保存协方差矩阵
        self.cov_all = [np.array([[1., 1., 1.], [1., 2., 1.], [1., 2., 1.]]),
                        np.array([[3., 1., 1.], [1., 4., 1.], [1., 2., 1.]]),
                        np.array([[2., 1., 1.], [1., 2., 1.], [1., 2., 1.]]),
                        np.array([[3., 1., 1.], [1., 4., 1.], [1., 2., 1.]])]  # 保存协方差矩阵
        self.u = np.array([[1., 2., 1.], [6., 7., 1.], [1., 2., 1.], [1., 2., 1.]])  # 均值
        self.alpha = [0.25, 0.25, 0.25, 0.25]  # 某一类的概率
        self.r = np.zeros((self.n, self.K))  # 响应度矩阵

    # 计算相似度（欧氏距离)
    @staticmethod
    def cal_similarity(lis1, lis2):
        x = np.mat(lis1) - np.mat(lis2)
        return np.dot(x, x.T)[0][0]

    # 计算损失
    def cal_loss(self):
        ans = 0
        for i in range(len(self.data)):
            for j in range(len(self.center)):
                # 每个点与每个中心计算欧式距离
                ans += self.cal_similarity(self.data[i, :], self.center[j, :])
        return ans

    # 选择初始中心
    def select_center(self):
        s = set()
        # 生成K个随机正整数
        while True:
            if len(s) == self.K:
                break
            s.add(int(math.fabs(random.randint(0, self.n - 1))))
        i = 0
        for num in s:
            self.center[i, 0] = self.data[num][0]
            self.center[i, 1] = self.data[num][1]
            i += 1

    # 更新各簇的中心点
    def new_center(self):
        for k in range(self.K):
            cnt = 0  # 计数
            a = np.zeros((1, self.dim))  # 统计和
            for i in range(self.n):
                if self.flags[i] == k:
                    cnt += 1
                    a += self.data[i, :]
            a /= cnt  # 均值
            self.center[k] = a

    # k-means算法
    def k_means(self):
        self.select_center()
        loss1 = self.cal_loss()
        while True:
            loss0 = loss1
            for i in range(self.n):
                flag = 0
                min_length = MAX_INT
                for j in range(self.K):
                    if min_length > self.cal_similarity(self.data[i, :], self.center[j, :]):
                        min_length = self.cal_similarity(self.data[i, :], self.center[j, :])
                        flag = j  # 寻找最近距离的类
                self.flags[i] = flag  # 重新标记
            self.new_center()  # 更新中心点
            loss1 = self.cal_loss()
            if np.fabs(loss1 - loss0) < DOOR:
                break

    # 可视化
    def draw(self, sig):
        x0 = []
        y0 = []
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        for i in range(self.n):
            x = self.flags[i]
            if sig == 1:
                x = self.flags_em[i]
            if x == 0:
                x0.append(self.data[i][0])
                y0.append(self.data[i][1])
            elif x == 1:
                x1.append(self.data[i][0])
                y1.append(self.data[i][1])
            elif x == 2:
                x2.append(self.data[i][0])
                y2.append(self.data[i][1])
            else:
                x3.append(self.data[i][0])
                y3.append(self.data[i][1])
        plt.legend(loc=2)
        plt.scatter(x0, y0, marker='*', c='orange', label='class1')
        plt.scatter(x1, y1, marker='o', c='lightskyblue', label='class2')
        plt.scatter(x2, y2, marker='o', c='lightgreen', label='class3')
        plt.scatter(x3, y3, marker='*', c='tomato', label='class4')
        plt.scatter(self.center[0, 0], self.center[0, 1], marker='+', color='blue', s=100, label='center')
        for k in range(1, self.K):
            plt.scatter(self.center[k, 0], self.center[k, 1], marker='+', color='blue', s=100)
        plt.xlabel('petal length')
        plt.ylabel('petal width')
        plt.legend(loc=2)
        plt.show()

    # 可视化
    def draw_3d(self, sig):
        x0 = []
        y0 = []
        z0 = []
        x1 = []
        y1 = []
        z1 = []
        x2 = []
        y2 = []
        z2 = []
        x3 = []
        y3 = []
        z3 = []
        for i in range(self.n):
            x = self.flags[i]
            if sig == 1:
                x = self.flags_em[i]
            if x == 0:
                x0.append(self.data[i][0])
                y0.append(self.data[i][1])
                z0.append(self.data[i][2])
            elif x == 1:
                x1.append(self.data[i][0])
                y1.append(self.data[i][1])
                z1.append(self.data[i][2])
            elif x == 2:
                x2.append(self.data[i][0])
                y2.append(self.data[i][1])
                z2.append(self.data[i][2])
            else:
                x3.append(self.data[i][0])
                y3.append(self.data[i][1])
                z3.append(self.data[i][2])
        ax = plt.subplot(111, projection='3d')
        plt.legend(loc=2)
        ax.scatter(x0, y0, z0, marker='*', c='orange', label='class1')
        ax.scatter(x1, y1, z1, marker='o', c='lightskyblue', label='class2')
        ax.scatter(x2, y2, z2, marker='o', c='lightgreen', label='class3')
        ax.scatter(x3, y3, z3, marker='*', c='tomato', label='class4')
        ax.scatter(self.center[0, 0], self.center[0, 1], self.center[0, 2], marker='+', color='blue', s=100,
                   label='center')
        for k in range(1, self.K):
            ax.scatter(self.center[k, 0], self.center[k, 1], self.center[k, 2], marker='+', color='blue', s=100)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend(loc=2)
        plt.show()

    # 计算初值，根据训练得到的k-means模型计算均值和协方差矩阵
    def cal_init(self):
        # 初始均值
        for i in range(self.K):
            for j in range(self.dim):
                self.u[i, j] = self.center[i, j]
        # 初始协方差
        cov = []
        for i in range(self.K):
            # 用每一类所有的样本计算每一类的协方差矩阵
            array = []
            for k in range(self.n):
                if self.flags[k] == i:
                    array.append(self.data[k, :])
            array = np.array(array)
            aver = []
            for k in range(len(array)):
                aver.append(self.u[i, :])
            aver = np.array(aver)
            cov.append(np.dot((array - aver).T, array - aver))
        self.cov_all = np.array(cov)

    # 计算概率密度 第j个样本属于第k类的概率
    def cal_probability(self, j, k):
        # 分母
        x1 = (2 * np.pi) ** (self.dim * 1.0 / 2) * np.linalg.det(self.cov_all[k]) ** 0.5
        # 分子
        x2 = np.exp(-0.5 * np.dot(np.dot((self.data[j, :] - self.u[k]), np.linalg.inv(self.cov_all[k])),
                                  (self.data[j, :] - self.u[k]).T))
        return x2 / x1
        # EM估计高斯混合模型的参数

    # 计算当为类k时，样本响应的概率和
    def cal_echo(self, k):
        ans = 0
        for i in range(self.n):
            ans += self.r[i][k]
        return ans

    def expectation_max(self, steps):
        # print('init cov:' + str(self.cov_all))
        # print('init aver:' + str(self.u))
        while steps > 0:
            steps -= 1
            # E步求每个样本属于每个类的响应度
            for j in range(self.n):
                alpha_multi_fi = 0
                for k in range(self.K):
                    # print(self.cal_probability(j, k))
                    alpha_multi_fi += self.alpha[k] * self.cal_probability(j, k)
                for k in range(self.K):
                    fi = self.cal_probability(j, k)
                    self.r[j][k] = self.alpha[k] * fi / alpha_multi_fi
            # M步更新参数
            for k in range(self.K):
                x1 = np.zeros((1, self.dim))  # 均值的分子
                for j in range(self.n):
                    x1 += self.r[j][k] * self.data[j, :]
                # 更新均值
                x2 = self.data - np.tile(self.u[k], (self.n, 1))  # 数据与均值的差值,np.tile(复制矩阵)
                x3 = np.eye(self.n)  # 每个样本对于类别k的概率对角阵
                for j in range(self.n):
                    x3[j][j] = self.r[j][k]
                    # 更新协方差矩阵
                self.cov_all[k] = np.dot(np.dot(x2.T, x3), x2) / self.cal_echo(k)
                # 更新alpha（每一类的概率）
                self.alpha[k] = self.cal_echo(k) / self.n
        print('cov:' + str(self.cov_all))
        print('aver:' + str(self.u))

    # 用最终得到的响应度进行分类
    def div_em(self):
        for j in range(self.n):
            rec = 0
            rate = 0
            for k in range(self.K):
                if self.r[j][k] > rate:
                    rec = k
                    rate = self.r[j][k]
            self.flags_em[j] = rec


def main():
    test = GaussMix()
    test.k_means()  # 先进行k-means训练
    test.draw(0)  # 画高斯分布得到的分类图
    test.cal_init()  # 用高斯分布得到的数据进行初始化
    test.expectation_max(100)  # EM算法进行参数估计
    test.div_em()  # 用高斯混合模型分类
    test.draw(1)  # 绘制分类图


def main_data():
    test = GaussMix()
    test.k_means()  # 先进行k-means训练
    test.draw_3d(0)  # 画高斯分布得到的分类图
    test.cal_init()  # 用高斯分布得到的数据进行初始化
    test.expectation_max(30)  # EM算法进行参数估计
    test.div_em()  # 用高斯混合模型分类
    test.draw_3d(1)  # 绘制分类图


# main()
main_data()
