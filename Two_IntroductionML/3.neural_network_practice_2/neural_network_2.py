# %%
# 目的:为深度学习领域的人们提供人工神经网络的介绍
# Keras 人工神经网络 基于Tensorflow和theano等低层神经网络API构建的高级神经网络API
# 人工神经网络 :非线性统计数据建模工具,对输入输出之间的复杂关系进行建模
# 激活函数:激活功能对于学习和理解输入和响应变量之间真正复杂的东西及非线性复杂的功能映射非常重要
# 激活函数的目的:将非线性特性引入到我们的网络中,目的:将ANN中的节点的输入信号转换成输出信号
# 反向传播:错误的向后传播,使用梯度下降法监督学习人工神经网络算法
# 给定一个人工神经网络和一个误差函数,该方法计算误差函数相对于神经网络权重的梯度


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#%%
# 导入数据
data = pd.read_csv("D:\\Python_project\\Jupyter_project\\Two_week\\neural_network_practice_2\\1_data\\data.csv")
#消除噪声 
del data['Unnamed: 32'] 

data.head(5)
#%%
# 拆分特征和标签
# df.iloc[行数,列数] 按位置筛选
# X 等于数据的所有行,2:n列的数据,标号 和id列数据不要
X = data.iloc[:,2:].values
# y输出是第一列diagnosis中的数据
y = data.iloc[:,1].values



#%%
# 对数据进行编码 预处理
# 从SKlearn预处理包中导入 标签编码模块
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
# 拟合标签编码和返回编码的标签
# 将字符标签转成数字的形式
y = labelencoder_X_1.fit_transform(y)
# 数据集分割
from sklearn.model_selection import train_test_split
# 测试数据集占10%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 0)
# 特征缩放
from sklearn.preprocessing import StandardScaler 
sc  = StandardScaler()
# 进行数据归一化处理
# fit_transform()的作用就是先拟合数据，然后转化它将其转化为标准形式。
X_train  = sc.fit_transform(X_train)
#  #已经找到了转换规则，我们把这个规则利用在训练集上，同样，我们可以直接将其运用到测试集
X_test  = sc.fit_transform(X_test)


#%%
# 准备好数据后,导入Keras 及其软件包
# Sequential:连续的,顺序的,有序的
# dense : 全连接层,下一个神经元节点的值是由上一层的所有节点计算得到的
# dropout: 随机失活层


# %%
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#%%
import keras 
from keras.models import Sequential
from keras.layers import Dense,Dropout

#%%
# 建立ANN模型
# 建立顺序模型
classifier = Sequential()
# 添加输入层和第一个隐藏层
#output_dim表示该层神经元是16个，'uniform'表示用均匀分布去初始化，activation代表激活函数，input_dim代表输入层特征个数
# classifier.add(Dense(16, activation= 'relu', input_dim= 30, use_bias= True)) 
classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30)) 
# 添加随机失活层防止过度拟合
#在隐藏层中删掉一些神经元，增加随机性，避免过拟合
classifier.add(Dropout(rate=0.1))

# input_dim-数据集的列数
# output_dim-要馈送到下一层的输出数（如果有）
# init-向ANN提供权重的方式



#%%
# 添加第二个隐藏层
classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
# 添加随机失活函数防止过拟合
classifier.add(Dropout(rate=0.1))


#%%
# 添加输出层
# output_dim为1，因为我们只希望最后一层的输出为1。

# Sigmoid函数用于处理两种类型结果的分类问题
# （Submax函数用于3种或更多分类结果）

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))


#%%
# compile 编译
# 编译ANN模型
# optimizer 优化方式选择
# loss:计算误差的方法选择
# metrics:评估方法选择
# adam :亚当
# ross-entropy loss:交叉熵损失（即对数损失）衡量分类模型的性能，该模型的输出是介于0和1之间的概率值。随着预测概率与实际标签的偏离，交叉熵损失会增加。因此，当实际观察标签为1时预测0.01的概率将很糟糕，并导致高损失值。理想模型的对数损失为0
# Binary_crossentropy是使用的损失函

from keras.optimizers import Adam

adam = Adam(learning_rate=0.01)
classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
#%%
# 拟合人工神经网络到训练数据集
classifier.fit(X_train,y_train,batch_size =100,epochs =150)
# 向前滚动很长，但值得
# 已通过反复试验设置了批次大小和历元数。仍在寻找更有效的方法。公开征求意见。


#%%
# batch:批次
# Batch size:批次大小定义了将通过网络传播的样本数量
# Epoch 训练次数


#%%
# 预测测试数据集的结果
y_pred = classifier.predict(X_test)
print(y_pred,"\n")
# 把输出结果变成真假的形式
y_pred = (y_pred > 0.5)
y_pred
#%%
# 构造混淆矩阵
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)


#%%
print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))
# %%
# heatmap()热度图
sns.heatmap(cm,annot=True)
plt.savefig('h.png')