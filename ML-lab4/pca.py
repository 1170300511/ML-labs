import numpy as np
from PIL import Image
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SIZE = 150  # 图片的大小
MAX = 255  # 图片的采样点为8，最大像素为255


def pca(data, kk):
    """pca降维函数"""

    # input:  data  矩阵 待降维数据集(mxn)
    #         kk     int  降维后的维数(kk<n)
    # output: data_rel  矩阵  降维后的矩阵(mxkk)
    #         mean      列表  均值，用于重构
    #         vt        矩阵  降维时右边乘的矩阵，用于重构

    """1.零均值化"""
    m = len(data)  # 样本数量
    n = len(data[0, :])  # 样本维数
    mean = []  # 每一维的均值
    for i in range(n):
        mean.append(sum(data[:, i] / m))
    data_aver = data - mean  # 对数据集的每一维数据进行零均值化

    """2.找到k个主成分特征向量，降维"""

    # svd特征值分解，其中
    # u为mxm的矩阵，
    # s^2相当于特征值分解的特征值（已经按从大到小的顺序排列），
    # d是对应的特征向量矩阵
    u, s, v = np.linalg.svd(data_aver)
    data_rel = np.dot(data_aver, v[:, :kk])  # 降维
    vt = v[:, :kk].T  # 重构时需要右乘的矩阵
    return mean, vt, data_rel


def generate_data(m):
    """生成3数据，并将生成的数据以矩阵形式返回"""

    # input:  m     int 生成的数据规模
    # output: data  矩阵 生成的数据，是一个mx3的矩阵

    data = np.zeros((m, 3))  # 初始化数据集
    mean1 = [1]  # 第一维的均值
    mean2 = [2]  # 第二维的均值
    mean3 = [3]  # 第三维的均值
    cov1 = [[200]]  # 第一维的方差
    cov2 = [[200]]  # 第二维的方差
    cov3 = [[200]]  # 第三维的方差，远小于一二维的方差
    data1 = np.random.multivariate_normal(mean1, cov1, m)  # 利用multivariate_normal生成三个高斯数据集
    data2 = np.random.multivariate_normal(mean2, cov2, m)
    data3 = np.random.multivariate_normal(mean3, cov3, m)
    for i in range(m):
        # 将每个维的数据对应到data中
        data[i][0] = data1[i]
        data[i][1] = data2[i]
        data[i][2] = data3[i]
    return data


def photo_pca(k):
    """人脸图片的pca降维"""
    # input: k int  k < SIZE

    data = read_photos()
    kk = k * k
    mean, vt, data_eig = pca(data, kk)  # 均值，右乘矩阵，降维后的矩阵
    recreate_photos(mean, vt, data, data_eig, SIZE, k)


def read_photos():
    """读取图片，生成样本数据"""

    # output: data 矩阵 每一行代表一张图片

    data = []
    for filename in os.listdir(r"./test"):  # 遍历文件夹中的所有图片
        pic = Image.open(r"./test/" + filename).convert('L')
        pic = pic.resize((SIZE, SIZE))  # 调整图片大小
        pic = np.array(pic).reshape((1, -1))[0].astype('float')  # 将每张图片转化为一维向量，并且将数据类型由uint8转变为double
        data.append(pic)
    # 缩小后的图片
    for i in range(len(data)):
        plt.imshow(Image.fromarray(data[i].reshape(SIZE, SIZE)))
        plt.axis('off')
        plt.show()
    return np.array(data)


def recreate_photos(mean, vt, data, data_eig, SIZE, k):
    """将降维后的数据重构成人脸"""
    data_eig = np.dot(data_eig, vt) + mean  # 重构后的图片矩阵

    m = len(data)
    n = len(data[0])
    for i in range(m):
        # 计算信噪比
        mse = 0
        for j in range(n):
            mse += (data[i][j] - data_eig[i][j]) ** 2 / n  # 计算第i张图的信噪比
        pnsr = 20 * math.log(MAX / math.sqrt(mse), 10)
        print(pnsr)
        pic = data_eig[i].reshape((SIZE, SIZE))  # 提取第i张图片的向量
        plt.imshow(Image.fromarray(pic))  # 画出降维后的图
        plt.title("k=" + str(k) + ", PNSR=" + str(pnsr))
        plt.axis('off')
        plt.show()


def train_data(m):
    """ 生成3维数据，降维为2维，并可视化"""

    data = generate_data(m)  # 生成3维数据
    *_, vt, data_eig = pca(data, 2)  # 将3维数据将为2维数据
    # 绘制图像
    fig = plt.figure()
    ax = Axes3D(fig)
    # 数据录入
    x = np.array(np.array(data[:, 0]))
    y = np.array(np.array(data[:, 1]))
    z = np.array(np.array(data[:, 2]))
    # 绘制降维前的散点图
    ax.scatter(x, y, z, marker='o', c='green', s=100)
    # 绘制降维的曲面
    f = np.cross(vt[0], vt[1])  # 根据基向量求投影面的法向量
    x_table, y_table = np.meshgrid(x, y)  # 数据表格化
    z_surf = -(f[0] * x_table + f[1] * y_table) / f[2]  # 根据法向量求z
    ax.plot_surface(x_table, y_table, z_surf, cmap='gray', shade=False, zorder=1)  # 绘制曲面
    # 绘制降维后的散点
    z_after = []
    for i in range(len(data)):
        z_after.append(-(f[0] * data[i][0] + f[1] * data[i][1]) / f[2])
    ax.scatter(x, y, z_after, marker='o', c='tomato', s=100, zorder=2)
    # 设置三个坐标轴信息
    ax.set_xlabel('x', color='black')  # 设置坐标轴
    ax.set_ylabel('y', color='black')
    ax.set_zlabel('z', color='black')

    plt.draw()
    plt.show()


if __name__ == '__main__':
    # train_data(10)  # 生成100个维数为3的数据，旋转并降维
    photo_pca(4)  # 读取图片并降维
