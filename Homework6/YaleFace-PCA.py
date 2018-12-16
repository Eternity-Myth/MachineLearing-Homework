import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# 数据的读取与初始化预处理
path = 'C:\\Users\\Eternity-Myth\\Desktop\\yalefaces'
for dirpath, subdir, file_set in os.walk(path):
    all_img = [path + '\\' + f for f in file_set]  # 保存所有文件的路径
m, n = len(all_img), len(misc.imread(all_img[0]).ravel())  # 行和列的数据
data = np.zeros((m, n))  # 初始化数据为（m,n）形状的矩阵
for i, f in enumerate(all_img):
    img = misc.imread(f).ravel()  # 将每个2D图像展平为1D阵列
    data[i] = img

# 对数据进行主成分分析（PCA）处理
data_centered = data - data.mean(axis=0)  # 对所有数据进行中心化
data_centered -= data_centered.mean(axis=1).reshape(m, -1)  # 对所有参数进行中心化
gap = data - data_centered  # 保存数据与中心化处理后的数据之间的关系
k = [20, 100]  # 保留的特征数k，设定k为20与100
pca1, pca2 = PCA(n_components=k[0]), PCA(n_components=k[1])
r_set, im_set = [], []  # 保存每个pca的方差比，输出去中心1D数组
for pca in [pca1, pca2]:
    lower_data = pca.fit_transform(data_centered)  # 形状是(166,k)
    comp = pca.components_  # 形状是(k,77760), 这是一个稀疏的二维数组
    r_set.append(np.sum(pca.explained_variance_ratio_))
    im_set.append(np.dot(lower_data, comp) + gap)

# 输出处理过后的数据图像
for j in range(1, 166):
    # 原图
    fig, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(10, 2.2))
    ax0.imshow(data[j].reshape((243, 320)), cmap=plt.cm.gray)
    ax0.set_title('origin')
    ax0.axis('off')
    # PCA降维后的图像
    for i, ax in enumerate([ax1, ax2]):
        ax.imshow(im_set[i][j].reshape((243, 320)), cmap=plt.cm.gray)
        ax.set_title('k=%s, Variance-Ratio: %.3f' % (k[i], r_set[i]))
        ax.axis('off')
    plt.subplots_adjust(left=0.02, bottom=0.05, right=0.98, wspace=0)
    plt.savefig(r'C:\\Users\\Eternity-Myth\\Desktop\\output\\' + str(j) + '.png')
