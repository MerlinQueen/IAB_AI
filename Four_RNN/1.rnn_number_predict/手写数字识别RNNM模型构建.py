# %%
from keras.layers import SimpleRNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
import keras
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

%matplotlib inline


sns.set(style='white', context='notebook', palette='deep')

print(tf.__version__)
# %%
# 加载数据
train = pd.read_csv(
    "D:\\Python_project\\Jupyter_project\\Three_week\\5.cnn_2\\subset_train.csv")
test = pd.read_csv(
    "D:\\Python_project\\Jupyter_project\\Three_week\\5.cnn_2\\Small_test.csv")


#  数据x,Y获取
Y_train = train['label']
X_train = train.drop('label', axis=1)

Y_test = test['label']
X_test = test.drop('label', axis=1)


g = sns.countplot(Y_train)

# %%
#  检查数据集中是否有空值
X_train.isnull().any().describe()
# %%
X_test.isnull().any().describe()

# %%
#  x数据归一化处理
X_train = (X_train - 0)/(255 - 0)
X_test = (X_test - 0)/(255-0)
# cnN.reshape(batch=-1 取所有,rows,cols,channels)
#  一维向量转二维
X_train = X_train.values.reshape(-1, 28, 28)  # 训练集合是4200个,28*28 通道数为1的输入
# RNN.reshape(batch_size,rows,cols)
X_test = X_test.values.reshape(-1, 28, 28)


# %%
# y独热值编码
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

# %%
#  训练数据集分割10折交叉验证
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=random_seed)

# %%
from keras.layers import SimpleRNN

batch_size = 100
num_classes = 10
epochs = 20
# 共有SEQLEN个字符
# 字典典大小是nb_chars
# input_shape = (SEQLEN,nb_chars)
input_shape = (28,28)
# 第二步搭建RNN模型
model = Sequential()  # 顺序结构创建

model.add(SimpleRNN(128, input_shape=input_shape, return_sequences=True))
model.add(Dropout(0.1))
# 如果RNN后面还是接RNN层,则应该return_sequences =true 返回所有时间戳
model.add(SimpleRNN(128,return_sequences =True))
model.add(Dropout(0.2))

model.add(SimpleRNN(128))
model.add(Dropout(0.2))

model.add(Dense(num_classes,activation='softmax'))
# 构建完毕,打印出来看看
model.summary()

# %%
#  设置模型优化器
# 优化器  尝试使用不同的优化器 至少以下三种
# 中文参考 https://keras.io/zh/optimizers/
##
## SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
## RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
## Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

optimizer_RMSprop = RMSprop(learning_rate=0.01, rho=0.9)
optimizer_Adam = Adam(learning_rate=0.01, beta_1=0.98, beta_2=0.98)
optimizer_SGD = SGD(learning_rate=0.01, momentum=0., nesterov=False)


# %%
#  模型编译
# 将模型compile 编译

model.compile(optimizer=optimizer_Adam,
              loss='categorical_crossentropy', metrics=['accuracy'])
# 自动优化
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=0.00001)


# %%
# 训练模型 
# 并保存
hitory = model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_val,Y_val))

model.save("D:\\Python_project\\Jupyter_project\\Four_week\\1.rnn_number_predict\\rnn_medel.h5")

# %%
# 生成学习曲线
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

# %%
# 生成10标签的混淆矩阵
def plot_confusion_matrix(cm,classes,normalize = False,title='confusion matrix',cmap = plt.cm.Blues):
    plt.imshow(cm,interpolation='r=nearest',cmap=cmap)
    