# %%
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import seaborn as sns
import keras
%matplotlib inline  

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.optimizers import RMSprop 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
sns.set(style = 'white',context = 'notebook',palette = 'deep')

print(tf.__version__)
# %%
# 第一步数据预处理
#  1. 加载数据
train  = pd.read_csv("D:\\Python_project\\Jupyter_project\\Three_week\\3.cnn_1\\train.csv")
test = pd.read_csv("D:\\Python_project\\Jupyter_project\\Three_week\\3.cnn_1\\test.csv")


train.head(5)
# %%
# 2.分割测试训练数据集的输入输出
Y_train = train['label']
# 删除 标签列得到训练数据集的输入
X_train = train.drop('label',axis = 1)
#   train数据处理完毕 , 释放内存
del train   
#   画出Y_train数据集中元素各值的个数
g= sns.countplot(Y_train)

#   用pd统计出Y-train中每个元素的个数
Y_train.value_counts() 




# %%
# 3.检查数据集中是否含有空值
X_train.isnull().any().describe()

test.isnull().any().describe()

# %%
#  4.图像压缩以及归一化处理
#   因为CNN在(0-1之间的收敛速度快于在0-255上),标准化把灰度图(0-255)映射到0-1区间    
X_train = X_train/255.
test = test/255.
#  把784个一维矢量,重构成RGB三通道3d矩阵
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# %%
#  5.对标签独热值编码0ne-hot
Y_train = to_categorical(Y_train,num_classes=10)

# %%
#  6.对训练数据集进行拆分训练数据集和验证数据集,10-fold拆分法则
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size =0.1,random_state =2)

g = plt.imshow(X_train[0][:,:,0])
# %%
#  第二步搭建CNN神经网络模型
#   1.分配样本数量,每一批的数量,设置迭代次数

batch_size = 64
num_classes = 10    
epochs = 20
#   输入数据的数量,输入size格式(长,宽,通道)
input_shape = (28,28,1)

#   2.利用序列累加Sequential构建神经网络模型
model = Sequential()
#   
#   构建输入层inut_shape,第一个卷积层32个kernel kernel 大小3*3,激活函数reLu kernel利用He_nomal正态分布随机生成    
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
#   构建第二个卷积层
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',kernel_initializer='he_normal'))

#   构建最大池化层,2*2,步长2
model.add(MaxPool2D((2,2),strides=2))

#   加入损失函数
model.add(Dropout(0.20))    

#   构建一个填充层padding 
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.20))

#   扁平化操作把模型拉伸成一维向量,作为全连接层的输入
model.add(Flatten())
#   构建一个全连接层,128个神经元
model.add(Dense(128, activation='relu'))
#############################################

# 模型训练优化
model.add(BatchNormalization())
model.add(Dropout(0.25))

#   构建全连接层输出层,用softmax函数激活,作为最后的输出
model.add(Dense(num_classes,activation='softmax'))





# %%
### 能否画出这个模型的概括图?
### 这个模型有几个卷积层？
### 这个模型最大的参数量是哪一层？
### 第一层卷积层为什么有320个实际变量需要调节
# 32*9+32+1  加上了一个偏置
#  打印出模型的摘要
model.summary()

# %%
