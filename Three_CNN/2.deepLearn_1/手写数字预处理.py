# %%
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg    
import seaborn as sns 
# 不用加plt.show()就能显示 魔法符号
%matplotlib inline  
# 导入数据集分割函数
from sklearn.model_selection import train_test_split
# 导入混淆矩阵评估函数
from sklearn.metrics import confusion_matrix 

import itertools
# 导入独热值编码函数
from keras.utils.np_utils import to_categorical

# 导入分类问题中的关于回归的函数
from keras.models import Sequential
# 导入神经网络层级建立函数
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
# 导入优化函数
from keras.optimizers import RMSprop
# 
from keras.preprocessing.image import ImageDataGenerator
# 
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')


print(tf.__version__)
# %%
# 读取数据
train = pd.read_csv("D:\\Python_project\\Jupyter_project\\Three_week\\2.deepLearn_1\\train.csv")
test = pd.read_csv("D:\\Python_project\\Jupyter_project\\Three_week\\2.deepLearn_1\\test.csv")
train.head(5)
# %%
# 利用pandas 的header 选择
Y_train = train['label']
# 因为train.csv中，第一列label在上述代码已经传递给Y_label，这里对于x_train 我们不需要训练集的第一列 #
X_train = train.drop(labels = ['label'],axis =1)
# 释放内存
del train 
g = sns.countplot(Y_train)
Y_train.value_counts()


# %%
# 检查训练数据是否有空值
X_train.isnull().any().describe()
# 检测测试数据集是否有空值
test.isnull().any().describe()
# %%
# 归一化处理
X_train = X_train/255.0
# 测试数据集归一化处理
test = test/255.0


# %%
#利用 reshape 函数， 将X_train变换成 (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# %%
# 特征编码
# 用0 1编码 将0-9数字标签编码成10维向量 (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)


# %%
# 数据集分割
# 设置随机种子
random_seed = 2
# 将训练集合按照9:1 分成训练集合 和验证集合 validation ####
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


# %%
# 例子
g = plt.imshow(X_train[0][:,:,0])

# %%
# 卷积神经网络

# %%
