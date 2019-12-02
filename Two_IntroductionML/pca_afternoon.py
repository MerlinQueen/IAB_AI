# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# 导入人脸识别数据集
from sklearn.datasets import fetch_lfw_people
# 导入sklearn中PCA函数
from sklearn.decomposition import PCA 

# %%
# 查看数据集的数据结构,
faces = fetch_lfw_people()
print(faces.data.shape)
# 把数据集用二维平面可视化展示
faces.images.shape
# %%
# 随机获取36张人脸,permutation序列,排序
np.random.seed(10)
random_indexs = np.random.permutation(len(faces.data))
X = faces.data[random_indexs]
data_faces = X[:36,:]
data_faces.shape
# %%
# 绘制这些人脸
def plot_digits(data):
    fig,axes = plt.subplots(6,6,figsize=(10,10),subplot_kw={'xticks':[],'yticks':[]},gridspec_kw=dict(hspace=0.1,wspace=0.1))
    for i,ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(62,47),cmap='bone')
 

plot_digits(data_faces)



# %%
# 读取脸对应的人名
faces.target_names
# 获取总人数
len(faces.target_names)
# %%
# 随机方式求取PCA
X,y = faces.data,faces.target
pca = PCA(svd_solver='auto')
pca.fit(X)

# %%
plot_digits(pca.components_)
# print(pca.components_.shape)


# %%
