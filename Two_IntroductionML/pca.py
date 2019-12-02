# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# %%
# axis = 0 按列取值
# axis = 1 按行取值
# 计算均值,输入数据为numpy的矩阵格式,行代表样本数,列表示特征
def meanX(dataA):
    return np.mean(dataA,axis=0)   #去数据表中的列数据求均值


# %%
# PCA 主成分分析
# Xmat:传入的是numpy格式的数据集矩阵
# k:表示要提取的前K个特征向量
def pca(Xmat,k ):
    """
    - finalData：參数一指的是返回的低维矩阵，相应于输入參数二
    - reconData：參数二相应的是移动坐标轴后的矩阵
    """
    # 第一步数据归一化处理(均值处理)
    # 求取平均数
    mean = meanX(Xmat)
    m,n = np.shape(Xmat)
    # 均值矩阵集,按纵列向下复制成与原始相同行数的矩阵
    means = np.tile(mean,(m,1))
    # 得到中心化(归一化)后的新矩阵
    data_adjust = Xmat - means 
    # 第二步求随机变量间的相关系(在各个维度上的相关性)
    # 求转置后的协方差(特征的协方差)
    # 因为特征向量是列向量,所以需要转置
    covrX = np.cov(data_adjust.T)
    # 第三步:求特征值和特征向量
    featureValue,featureVector = np.linalg(covrX)
    # 把特征值从大到小排序
    # np.argsort(A),把矩阵中的元素从小到大排序,并返回对应的索引
    index = np.argsort(-featureValue)
    # 如果目标维度大于n就是升维,不是降维了
    if k>n:
        print("k must lower than feature number")
        return
    else:
        # 选择前K个特征值对应的特征向量
        # 因为前面求协方差进行了一次转置,再转一次还原
        # 最后一步数据还原,得到降维后的数据
        selectVec = np.matrix(featureVector.T[index[:k]])
        finalData = data_adjust*selectVec.T
        reconData = (finalData*selectVec)+ mean
    return finalData,reconData


# %%
#输入文件的每行数据都以\t隔开
def loaddata(datafile):
    return np.array(pd.read_csv(datafile,sep="\t",header=-1)).astype(np.float)


# %%
# 主成分分析的Python模块实现
from sklearn.decomposition import PCA
inputfile = "data.xls"
outputfile = ""