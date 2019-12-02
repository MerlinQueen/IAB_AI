# %%
import numpy as np 
import pandas as  pd 
import os


#%%
# 读入hour.csv文件到raw 中
raw = pd.DataFrame(pd.read_csv(("D:\\Python_project\\Jupyter_project\\Two_week\\3.neural_network_practice_2\\1_data\\bike-sharing-dataset\\hour.csv")))
raw.head()
raw.tail()

#%%
# 获取每一列的标题索引
raw.columns
# 查看数据的均值,最大值,最小值等描述
raw.describe()
#%%
# 进行编码,把标签变量转成虚拟变量
def generate_dummies(df,dummy_column):
    dummies = pd.get_dummies(df[dummy_column],prefix = dummy_column)
    # 把变换后的特征连接到df表格之后
    df = pd.concat([df,dummies],axis=1)
    return df 

X = pd.DataFrame.copy(raw)
dummy_columns = ['season','yr','mnth','hr','weekday','weathersit']
for dummy_column in dummy_columns:
    X = generate_dummies(X,dummy_column)
X.head()
X.columns
#%%
# 删除X 中的分类变量
for dummy_column in dummy_columns:
    del X[dummy_column]
X.columns
X.head(5)
# %%
# 对X中前3周数据画图,x轴dteday,y轴cnt,大小(18,5)
import matplotlib.pyplot as plt 
# 3周总共的小时数
first_3_weeks = 3*7*24
X[:first_3_weeks].plot(x='dteday',y='cnt',figsize =(18,5))
plt.show()

#%%
# 去掉dteday列和instans列
del X['dteday']
del X['instant']
# %%
#  设置目标标签列,删除多于标签
y = X['cnt']
del X['cnt']
del X['registered']
del X['casual']

#%%
X.head()
# 数据分割
# 获取包含一整天的数据
all_days = len(X)//24
print("total observations",len(X))
print("total number of days",all_days)
# 取总样本的百分之70作为训练数据集
days_for_training = int(len(X)*0.7)
# X_train赋值为X中的前days_for_training行数据
X_train = X[0:days_for_training] 
# X_test赋值为其余数据
X_test = X[days_for_training:]
#%%
print("训练数据集样本数为", len(X_train))
print("测试数据集的样本为", len(X_test))
print("目标标签值为", y.head())


#%%
# 把目标值归一化
# 使用最大最小值归一化方法
y_normalized = (y-y.min())/(y.max()-y.min())

y_normalized.head(5)
y_train = y[:days_for_training]
y_test = y[days_for_training:]
y_train_normalized = y_normalized[0:days_for_training]
y_test_normalized = y_normalized[days_for_training:]

#%%
# 建立一个简单的模型,
from keras.models import Sequential
from keras.layers import Dense,Dropout
# 获取特征个数
features = X.shape[1]
print('features is \n',features)


#%%
# 顺序模型,一种前馈网络模型
model = Sequential()
# 建立网络模型
# 添加输入层和第一层
# dense参数选项
# units:该层有几个神经元
# activation: 该层使用的激活函数
# use_bias:是否添加偏置项
# kernel_initializer:权重初始化方法
# bias_initializer:偏置值初始化方法
# kernel_regularizer:权重规范化函数
# bias_regularizer:偏置值规范化方法
# # activity_regularizer:输出的规范化方法
# kernel_constraint:权重变化限制函数
# bias_constraint:偏置值变化限制函数
# 制定第一层时需要制定数据的输入形状即,input_dim
# input_shape就是指输入张量的shape。例如，input_dim=784，说明输入是一个784维的向量，这相当于一个一阶的张量，它的shape就是(784,)。因此，input_shape=(784,)。

# 隐藏层的第一层神经元13个,输入层59个
# 第一层添加方式一
model.add(Dense(5,input_shape=(features,),activation='relu'))
# 第一层添加方式二
# input_dim = input_shape(features,)
# model.add(Dense(13,input_dim = imput_dim,activation='relu'))
# 添加损失函数
# 添加第二层隐藏层

model.add(Dense(4,activation='relu'))
model.add(Dropout(0.01))
# 添加输出层
model.add(Dense(1,activation='linear'))
# 显示模型概述,摘要
model.summary()

#%%
# 模型编译,错误率0.01,采用sigmoid算法
from keras.optimizers import SGD
sgd = SGD(0.01)
model.compile(optimizer=sgd, loss="mean_squared_error")


#%%
# 模型训练
# validation:确认
results = model.fit(X_train, y_train_normalized, epochs=150,validation_data = (X_test, y_test_normalized))
# %%
# 打印结果并画图
results.history
pd.DataFrame.from_dict(results.history).plot()
# %%
# 模型评估
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

y_test_normalized=np.array(y_test_normalized)
y_pred = model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test_normalized, y_pred))
r2 = r2_score(y_test_normalized, y_pred, multioutput='raw_values')

# rmse越小越好
# R2越大越好
print("RMSE:",rmse)
print("R2:",r2)