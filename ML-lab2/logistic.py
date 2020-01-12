import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import random

"""
by xiaoyi
"""


class Logistic:
    matrix = ()  # 读入数据矩阵
    test_matrix = ()  # 测试数据矩阵
    y = ()  # 分类情况，y[i]表示第i组数据的分类情况
    test_y = ()  # 测试数据集的分类情况
    x = ()  # 特征矩阵，其中x[i]表示第i个实例的特征取值情况,最后一维为1
    test_x = ()  # 测试数据集的特征矩阵
    w = ()  # 对应扩充特征后的w
    n = 0  # 特征数的个数，其中w是n+1维的
    dataSum = 0  # 数据量
    testSum = 0  # 测试数据集大小

    # sigmoid函数
    @staticmethod
    def sig(wx):
        if wx < -10:
            return 0
        else:
            return 1 / (1 + math.exp(-wx))

    # 计算对数似然的值，不加正则，梯度上升法，没有加负号
    def cal_loss1(self):
        loss = 0
        for i in range(self.dataSum):
            w_multi_x = np.dot(self.x[i], self.w)
            # print(w_multi_x)
            loss -= np.dot(self.y[i], w_multi_x)
            # 防止溢出，所以对wx进行讨论
            if w_multi_x > 0:
                loss += w_multi_x + math.log(1 + math.exp(-w_multi_x))
            else:
                loss += math.log(1 + math.exp(w_multi_x))
        return loss

    # 计算损失函数的值，加正则，梯度下降法，加负号
    def cal_loss2(self, regex):
        loss = 0
        for i in range(self.dataSum):
            # print(self.x[i])
            w_multi_x = np.dot(np.mat(self.x[i]), self.w)
            # print(w_multi_x)
            loss -= np.dot(self.y[i], w_multi_x)
            # 防止溢出，所以对wx进行讨论
            if w_multi_x > 0:
                loss += w_multi_x + math.log(1 + math.exp(-w_multi_x))
            else:
                loss += math.log(1 + math.exp(w_multi_x))
        loss += regex * np.dot(self.w.T, self.w)[0, 0]
        # loss /= self.dataSum
        return loss

    """
    # 计算梯度，不带正则，似然函数的梯度，批量梯度（计算出现问题，暂时还未解决）
    def cal_gradient1(self):
        gradient = np.zeros((self.n + 1, 1))
        for i in range(self.dataSum):
            w_multi_x = np.dot(self.w.T, self.x[i].T)
            eg = self.y[i] * np.eye(self.n + 1)
            gradient -= np.dot(np.mat(self.x[i]), eg).T
            if w_multi_x >= 0:
                eg = math.exp(1 / (1 + math.exp(-w_multi_x))) * np.eye(self.n + 1)
                gradient += np.dot(eg, self.w)
            else:
                if math.exp(w_multi_x) >= 1e-20:
                    eg = math.exp(w_multi_x) / (1 + math.exp(w_multi_x)) * np.eye(self.n + 1)
                    gradient += np.dot(eg, self.w)
        return gradient
    """

    # 计算梯度 ，随机下降法
    def cal_gradient1(self):
        gradient = np.zeros((self.n + 1, 1))
        i = random.randint(0, self.dataSum - 1)
        wx = np.dot(np.mat(self.x[i]), self.w)
        for j in range(self.n + 1):
            gradient[j][0] += self.x[i][j] * (-self.y[i] + Logistic.sig(wx))
        return gradient

    # 计算梯度，带正则，损失函数的梯度
    def cal_gradient2(self, regex):
        gradient = np.zeros((self.n + 1, 1))
        i = random.randint(0, self.dataSum - 1)
        wx = np.dot(np.mat(self.x[i]), self.w)
        for j in range(self.n + 1):
            gradient[j][0] += self.x[i][j] * (-self.y[i] + Logistic.sig(wx))
        gradient += regex * self.w
        # print(gradient)
        # gradient /= self.dataSum
        # print(gradient)
        return gradient

    # 使用梯度下降法优化参数，似然函数，不带正则
    def de_gradient1(self, lamda, door):
        # print(self.w)
        loss0 = self.cal_loss1()
        g0 = self.cal_gradient1()
        w0 = self.w
        self.w -= lamda * g0
        loss1 = self.cal_loss1()
        cnt = 0
        while cnt < door:
            cnt += 1
            loss0 = loss1
            g0 = self.cal_gradient1()
            w0 = self.w
            self.w -= lamda * g0
            loss1 = self.cal_loss1()
            # print(loss0 - loss1)
        self.w = w0
        # print(self.w)
        # 返回损失函数的值
        return loss0

    # 使用梯度下降法求解带正则项的w
    def de_gradient2(self, lamda, door, regex):
        loss0 = self.cal_loss2(regex)
        g0 = self.cal_gradient2(regex)
        w0 = self.w
        self.w -= lamda * g0
        loss1 = self.cal_loss2(regex)
        cnt = 0
        while cnt < door:
            # print(loss1 - loss0)
            # print(g0)
            cnt += 1
            loss0 = loss1
            g0 = self.cal_gradient2(regex)
            w0 = self.w
            self.w -= lamda * g0
            loss1 = self.cal_loss2(regex)
        self.w = w0
        # 返回损失函数的值
        return loss0

    # 计算黑塞矩阵
    def hessian(self):
        he = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.dataSum):
            w_multi_x = np.dot(np.mat(self.x[i]), self.w)
            # print(w_multi_x)
            for j in range(self.n + 1):
                for k in range(self.n + 1):
                    if w_multi_x > 20:
                        he[j][k] -= 0
                    else:
                        p = Logistic.sig(w_multi_x)
                        he[j][k] += self.x[i][j] * self.x[i][k] * p * (1 - p)
        return he

    # 牛顿法
    def newton(self, steps):
        cnt = 0
        w0 = self.w
        while cnt < steps:
            cnt += 1
            g = self.cal_gradient1()
            # print(g)
            he = self.hessian()
            # print(np.linalg.inv(he))
            w0 = self.w
            # print(self.w)
            self.w -= np.dot(np.linalg.inv(he), g)
        self.w = w0

    # 读取训练集
    def read_data(self, file):
        self.matrix = pd.read_csv(file, header=1).values
        # print(self.matrix)
        # with open(file) as f:
        #    self.matrix = np.loadtxt(f, float, delimiter=",")
        self.dataSum = len(self.matrix)
        self.n = len(self.matrix[0]) - 1
        add = np.ones((self.dataSum, 1))
        self.x = np.hstack((self.matrix[:, :self.n], add))
        # print(self.x)
        self.y = self.matrix[:, self.n]
        self.w = np.ones((self.n + 1, 1))

    # 读取测试集
    def read_test_data(self, file):
        self.test_matrix = pd.read_csv(file, header=1).values
        # with open(file) as f:
        #    self.test_matrix = np.loadtxt(f, float, delimiter=',')
        self.testSum = len(self.test_matrix)
        self.test_x = np.hstack((self.test_matrix[:, :self.n], np.ones((self.testSum, 1))))
        self.test_y = self.test_matrix[:, self.n]

    # 预测
    def pre_test(self):
        cnt = 0
        for i in range(self.testSum):
            pre_wx = np.dot(np.mat(self.test_x[i]), self.w)
            # print(pre_wx)
            if (pre_wx >= 0) and (self.test_y[i] == 1):
                cnt += 1
            elif (pre_wx <= 0) and (self.test_y[i] == 0):
                cnt += 1
        return cnt / self.testSum


def test_model():
    # 测试模型
    test = Logistic()
    train_set = "gauss.csv"
    test_set = "test_gauss.csv"
    test.read_data(train_set)
    lamda = 1e-2
    steps = 10
    regex = 1e-3
    # test.de_gradient2(lamda, steps, regex)
    # test.de_gradient1(lamda, steps)
    test.newton(steps)
    test.read_test_data(test_set)
    correct = test.pre_test()
    print(correct)
    x0 = test.test_matrix[:500, 0]
    y0 = test.test_matrix[:500, 1]
    x1 = test.test_matrix[500:, 0]
    y1 = test.test_matrix[500:, 1]
    plt.scatter(x0, y0, marker='.', color='lightgreen')
    plt.scatter(x1, y1, marker='+', color='lightskyblue')
    dx = np.linspace(0, 10, 100)
    dy = (-test.w[2][0] - test.w[0][0] * dx) / test.w[1][0]
    # plt.title("lamda=" + str(lamda) + ",steps=" + str(steps)+",regex ="+str(regex))
    # plt.title("lamda=" + str(lamda) + ",steps=" + str(steps))
    plt.plot(dx, dy, color='y')
    ans = "shot rate= " + str(correct)
    plt.text(0, 1, ans, color='hotpink', fontsize=15)
    plt.show()



def generate_data():
    # 生成高斯数据
    f = open('test_gauss_not_bayes.csv', 'w')
    mean0 = [2, 3]
    cov = np.mat([[2, 1], [1, 2]])
    x0 = np.random.multivariate_normal(mean0, cov, 500).T

    mean1 = [7, 8]
    x1 = np.random.multivariate_normal(mean1, cov, 500).T

    for i in range(len(x0.T)):
        line = []
        line.append(x0[0][i])
        line.append(x0[1][i])
        line.append(1)
        line = ",".join(str(i) for i in line)
        line = line + "\n"
        f.write(line)

    for i in range(len(x0.T)):
        line = []
        line.append(x1[0][i])
        line.append(x1[1][i])
        line.append(0)
        line = ",".join(str(i) for i in line)
        line += "\n"
        f.write(line)
    f.close()


test_model()
