# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
%matplotlib  inline

print(tf.__version__)
# %%
#  设置你想要的帧是斜率和截距,以及数据点(50-200)
true_w = 5
true_b = 3
num_samples = 200

# 初始化随机数据

X = tf.random.normal(shape=[num_samples, 1]).numpy()
noise = tf.random.normal(shape=[num_samples, 1]).numpy()
#  添加噪声
y = X*true_w+true_b+noise

# 画出散点图
plt.scatter(X,y)

# %%
# 随机初始化参数
# uiform 均匀分布
W = tf.Variable(tf.random.uniform([1]))
b = tf.Variable(tf.random.uniform([1]))
# 定义一个随机一元一次线性函数
# x小写:函数的一个变量,形参
def random_line(x):
    y = W * x + b
    return y

plt.scatter(X,y)
plt.plot(X,random_line(X),c='r')

# %%
# 损失函数
def loss_fn(x, y):
    y_ = random_line(x)
    return tf.reduce_mean(tf.square(y_- y))

# %%
# 通过改变epochs迭代次数的值,和Learning rate学习率,步长观察梯度下降函学习的线性函数w,b的值以及loss函数的变换
EPOCHS = 10
LEARNING_RATE  = 0.1



for epoch in range(EPOCHS):  # 迭代次数
    with tf.GradientTape(persistent=True) as tape:  # 追踪梯度
        loss = loss_fn(X, y)  # 计算损失
    dW, db = tape.gradient(loss,[W, b])  # 计算梯度
    W.assign_sub(LEARNING_RATE * dW)  # 更新梯度
    b.assign_sub(LEARNING_RATE * db)
    # 输出计算过程
    print('Epoch [{}/{}], loss [{:.3f}], W is [{:.3f}], b is [{:.3f}]'.format(epoch, EPOCHS, loss,
                                                                     float(W.numpy()),
                                                                     float(b.numpy())))
# %%
