import numpy as np
from numpy import *
import matplotlib.pyplot as plt

np.seterr(divide="ignore", invalid='ignore')


# 求解矩阵X
def calX(dataAmount, n, dataMatrix):
    X = ones((dataAmount, n + 1))  # 初始化
    for i in range(dataAmount):
        for j in range(n):
            X[i][j] = dataMatrix[i][0] ** (j + 1)
        X[i][n] = 1
    return X


# 计算拟合的函数值
def cal(x, n, w):
    rel = 0
    for j in range(n):
        rel += x ** (j + 1) * w[j]
    rel += w[n]
    return rel


# 计算loss(不带惩罚项)
def calLoss(w, y, X, dataAmount):
    rel = y - np.dot(X, w)
    return np.dot(rel.T, rel) / (2 * dataAmount)


# 不带正则项的解析解
def analysis(dataAmount, n):
    doc = str(dataAmount) + ".txt"
    dataMatrix = np.loadtxt(doc, dtype=float)  # 样例的读入矩阵
    cols = dataMatrix.shape[-1]
    y = dataMatrix[:, cols - 1:cols]  # 样例的y值
    X = calX(dataAmount, n, dataMatrix)
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    loss = calLoss(w, y, X, dataAmount)

    dx = []
    dx1 = []
    d = 2 * np.pi / (dataAmount * 10)
    cnt = 0
    for i in range(dataAmount):
        dx1.append(dataMatrix[i][0])
    for i in range(dataAmount * 10):
        dx.append(cnt)
        cnt += d
    dy = []
    dy1 = []
    for i in range(dataAmount):
        dy1.append(dataMatrix[i][1])
    for i in range(len(dx)):
        dy.append(cal(dx[i], n, w))
    plt.plot(dx, dy, 'r')  # 拟合函数
    plt.plot(dx, sin(dx), 'b')
    plt.scatter(dx1, dy1, c="#000000", marker='.')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(str(n) + "-order to fit sinx of data-" + str(dataAmount))
    plt.show()

    return loss[0][0]


# analysis(40,7)

# 带正则项的解析解
def analysis_regex(dataAmount, n, lamda):
    doc = str(dataAmount) + ".txt"
    dataMatrix = np.loadtxt(doc, dtype=float)
    cols = dataMatrix.shape[-1]
    y = dataMatrix[:, cols - 1:cols]
    X = calX(dataAmount, n, dataMatrix)
    I = eye(n + 1)
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lamda * dataAmount * I), X.T), y)
    loss1 = calLoss(w, y, X, dataAmount)  # 对比不同lamda的损失
    # loss = calLoss(w, y, X, dataAmount) + lamda*np.dot(w.T, w)/2
    """
    dx = []
    dx1 = []
    d = 2 * np.pi / (dataAmount * 10)
    cnt = 0
    for i in range(dataAmount):
        dx1.append(dataMatrix[i][0])
    for i in range(dataAmount * 10):
        dx.append(cnt)
        cnt += d
    dy = []
    dy1 = []
    for i in range(dataAmount):
        dy1.append(dataMatrix[i][1])
    for i in range(len(dx)):
        dy.append(cal(dx[i], n, w))
    plt.plot(dx, dy, 'r')
    plt.plot(dx, sin(dx), 'b')
    plt.scatter(dx1, dy1, c="#000000", marker='.')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(str(n) + "-order to fit sinx of data-"+str(dataAmount))
    plt.show()
    """
    return loss1[0][0]


# analysis_regex(20,8,0.001)

# 求梯度
def gradient(X, w, y, dataAmount):
    return np.dot(X.T, np.dot(X, w) - y) / dataAmount


# 梯度下降法

def de_gradient(dataAmount, n, lamda):
    doc = str(dataAmount) + ".txt"
    dataMatrix = np.loadtxt(doc, dtype=float)
    cols = dataMatrix.shape[-1]
    y = dataMatrix[:, cols - 1:cols]
    X = calX(dataAmount, n, dataMatrix)
    w = 0.1 * ones((n + 1, 1))
    on = 1e-7
    g0 = gradient(X, w, y, dataAmount)
    # cnt = 5000000
    while 1:
        loss = calLoss(w, y, X, dataAmount)
        w += (-lamda) * g0
        print(g0)
        loss1 = calLoss(w, y, X, dataAmount)
        g = gradient(X, w, y, dataAmount)
        # cnt -= 1
        if abs(loss - loss1) < on:
            break
        g0 = g
    dx = []
    dx1 = []
    d = 1 / (dataAmount * 10)
    cnt = 0
    for i in range(dataAmount):
        dx1.append(dataMatrix[i][0])
    for i in range(dataAmount * 10):
        dx.append(cnt)
        cnt += d
    dy = []
    dy1 = []
    dy2 = []
    for i in range(dataAmount):
        dy1.append(dataMatrix[i][1])
    for i in range(len(dx)):
        dy.append(cal(dx[i], n, w))
        dy2.append(sin(2 * np.pi * dx[i]))
    plt.plot(dx, dy, 'r')
    plt.plot(dx, dy2, 'b')
    plt.scatter(dx1, dy1, c="#000000", marker='.')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(str(n) + "-order to fit sinx of data-" + str(dataAmount))
    plt.show()

    loss = calLoss(w, y, X, dataAmount)
    return loss[0][0]


de_gradient(10, 4, 0.0001)

"""
#梯度下降法
def de_gradient(dataAmount,n,lamda):
    doc = str(dataAmount) + ".txt"
    dataMatrix = np.loadtxt(doc, dtype=float)
    cols = dataMatrix.shape[-1]
    y = dataMatrix[:, cols-1:cols]
    X = calX(dataAmount, n, dataMatrix)
    w = 0.1*ones((n+1, 1))
    on = 1e-10
    loss = calLoss(w, y, X, dataAmount)
    g0 = gradient(X, w, y, dataAmount)
    dg = np.eye(n+1)
    while 1:
        #print(g0)
        w1 = w + (-lamda) * np.dot(dg, g0)
        loss1 = calLoss(w1, y, X, dataAmount)
        g1 = gradient(X, w1, y, dataAmount)
        w2 = w1 + (-lamda) * np.dot(dg, g1)
        for i in range(n+1):
            if (w[i][0]-w1[i][0])*(w1[i][0] - w2[i][0]) < 0:
                dg[i][i] = dg[i][i] / 10
        if abs(loss[0][0]-loss1[0][0]) < on:
            break
        else:
            w = w + (-lamda) * np.dot(dg, g0)
            loss = calLoss(w, y, X, dataAmount)
            g0 = gradient(X, w, y, dataAmount)
    dx = []
    dx1 = []
    d = 1 / (dataAmount * 10)
    cnt = 0
    for i in range(dataAmount):
        dx1.append(dataMatrix[i][0])
    for i in range(dataAmount * 10):
        dx.append(cnt)
        cnt += d
    dy = []
    dy1 = []
    dy2 = []
    for i in range(dataAmount):
        dy1.append(dataMatrix[i][1])
    for i in range(len(dx)):
        dy.append(cal(dx[i], n, w))
        dy2.append(sin(2*np.pi*dx[i]))
    plt.plot(dx, dy, 'r')
    plt.plot(dx, dy2, 'b')
    plt.scatter(dx1, dy1, c="#000000", marker='.')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(str(n) + "-order to fit sinx of data-"+str(dataAmount))
    plt.show()

    loss = calLoss(w, y, X, dataAmount)
    return loss[0][0]
de_gradient(10,3,0.001)
"""


# 共轭梯度法
def cj_gradient(dataAmount, n):
    doc = str(dataAmount) + ".txt"
    dataMatrix = np.loadtxt(doc, dtype=float)
    cols = dataMatrix.shape[-1]
    y = dataMatrix[:, cols - 1:cols]
    X = calX(dataAmount, n, dataMatrix)
    A = np.dot(X.T, X)
    w = zeros((n + 1, 1))
    b = np.dot(X.T, y)
    r = b - np.dot(A, w)
    p = r
    while 1:
        if np.dot(r.T, r) == 0 or np.dot(np.dot(p.T, A), p) == 0:
            break
        a = np.dot(r.T, r) / np.dot(np.dot(p.T, A), p)
        w += a * p
        r1 = r - np.dot(a * A, p)
        beta = np.dot(r1.T, r1) / np.dot(r.T, r)
        p = r1 + beta * p
        r = r1
    loss = calLoss(w, y, X, dataAmount)

    dx = []
    dx1 = []
    d = 2 * np.pi / (dataAmount * 10)
    cnt = 0
    for i in range(dataAmount):
        dx1.append(dataMatrix[i][0])
    for i in range(dataAmount * 10):
        dx.append(cnt)
        cnt += d
    dy = []
    dy1 = []
    for i in range(dataAmount):
        dy1.append(dataMatrix[i][1])
    for i in range(len(dx)):
        dy.append(cal(dx[i], n, w))

    plt.plot(dx, dy, 'r')
    plt.plot(dx, sin(dx), 'b')
    plt.scatter(dx1, dy1, c="#000000", marker='.')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(str(n) + "-order to fit sinx of data-" + str(dataAmount))
    plt.show()

    return loss[0][0]


# cj_gradient(40,7)

# 比较数据量为20时不同阶不带正则项解析解的损失
def main1(max_n):
    dataAmount = 20
    x = []
    for i in range(1, max_n):
        x.append(i)
    y = []
    for i in range(1, max_n):
        y.append(analysis(dataAmount, x[i - 1]))
    plt.plot(x, y, 'b')
    plt.title("different order's loss")
    plt.xlabel("order")
    plt.ylabel("loss")
    plt.show()


# main1(10)

# 比较数据量为20时不同阶带正则项解析解的损失
def main2(max_n):
    dataAmount = 20
    x = []
    for i in range(1, max_n):
        x.append(i)
    y = []
    for i in range(1, max_n):
        y.append(analysis_regex(dataAmount, x[i - 1], 0.0001))
    plt.plot(x, y, 'b')
    plt.show()


# main2(10)

# 比较数据量为20时不同阶的梯度下降法的损失
def main3(max_n):
    dataAmount = 20
    lamda = 0.0001
    x = []
    for i in range(1, max_n):
        x.append(i)
    y = []
    for i in range(1, max_n):
        a = 10 ^ i
        y.append(de_gradient(dataAmount, i, lamda / a))
    plt.plot(x, y, 'b')
    plt.show()


# main3(10)

# 比较数据量为20时不同阶的共轭梯度法的损失
def main4(max_n):
    dataAmount = 20
    x = []
    for i in range(1, max_n):
        x.append(i)
    y = []
    for i in range(1, max_n):
        y.append(cj_gradient(dataAmount, i))
    plt.plot(x, y, 'b')
    plt.xlabel("order")
    plt.ylabel("loss")
    plt.title("different order's loss")
    plt.show()


# main4(10)

# 比较阶数为8时，不同数据量的表现
def main_data1():
    x = []
    for i in range(10, 100, 10):
        x.append(i)
    y = []
    for i in range(len(x)):
        y.append(analysis(x[i], 8))
    plt.plot(x, y, 'b')
    plt.title("different data's loss")
    plt.xlabel("dataAmount")
    plt.ylabel("loss")
    plt.show()


# main_data1()

# 比较数据量为20时带正则项和不带正则项
def main_data20_regexandnot(lamda, n):
    dataAmount = 10  # 数据量
    I = eye(n + 1)
    doc = str(dataAmount) + ".txt"
    dataMatrix = np.loadtxt(doc, dtype=float)
    cols = dataMatrix.shape[-1]
    y = dataMatrix[:, cols - 1:cols]
    X = calX(dataAmount, n, dataMatrix)
    # w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    w_regex = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lamda * dataAmount * I), X.T), y)
    # loss = calLoss(w, y, X, dataAmount)

    dx = []
    dx1 = []
    d = 2 * np.pi / (dataAmount * 10)
    cnt = 0
    for i in range(dataAmount):
        dx1.append(dataMatrix[i][0])
    for i in range(dataAmount * 10):
        dx.append(cnt)
        cnt += d
    dy = []
    dy1 = []
    dy_regex = []
    for i in range(dataAmount):
        dy1.append(dataMatrix[i][1])
    print(w_regex)
    for i in range(len(dx)):
        # dy.append(cal(dx[i], n, w))
        dy_regex.append(cal(dx[i], n, w_regex))
    # plt.plot(dx, dy, 'r')
    plt.plot(dx, dy_regex, 'r')
    plt.plot(dx, sin(dx), 'b')
    plt.legend(['with no regex', 'sinx'])
    plt.scatter(dx1, dy1, c="#000000", marker='.')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(str(n) + "-order to fit sinx of data-" + str(dataAmount) + ",lamda = " + str(lamda))
    plt.show()


# main_data20_regexandnot(0.05,9)

# 研究不同lamda正则因子下的loss，最好关掉正则化函数的plot
def different_lamda():
    lamda = []
    y = []
    d = 1e-11
    for i in range(10000):
        lamda.append(d * i)
    for i in range(10000):
        y.append(analysis_regex(20, 8, lamda[i]))
    plt.plot(lamda, y)
    plt.title("loss of differnt lamda")
    plt.xlabel("lamda")
    plt.ylabel("loss")
    plt.show()
# different_lamda()
