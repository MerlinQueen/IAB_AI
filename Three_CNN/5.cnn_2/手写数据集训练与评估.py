# %%
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
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

#  一维向量转二维
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)


# %%
# y独热值编码
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

# %%
#  训练数据集分割10折交叉验证
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=random_seed)
#  随机显示一个样本
g = plt.imshow(X_train[0][:, :, 0], cmap='gray')

# %%
# 第二步搭建神经网络模型
# CNN
batch_size = 40
num_classes = 10
epochs = 25
input_shape = (28, 28, 1)
# 构建模型
model = Sequential()
# 第一层卷积层
model.add(Conv2D(32, (3, 3), activation='relu',
                 kernel_initializer='he_normal', input_shape=input_shape))
# 池化层
model.add(MaxPool2D((2, 2), strides=2))

# 第二层卷积层
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'))
# 第二层池化
model.add(MaxPool2D((2, 2), strides=1))
# 随机失活层
model.add(Dropout(0.25))
# 扁平化层,变成一维向量
model.add(Flatten())
# 全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(80, activation='relu'))
# 批次归一化层
model.add(BatchNormalization())
model.add(Dropout(0.15))
# 输出层
model.add(Dense(num_classes, activation='softmax'))

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
# 调节loss 参数，即
# loss function损失函数选择:
#   1. mean_squared_error
#   2. categorical_crossentropy/为什么不用binary_crossentropy
#   3. mean_absolute_error

model.compile(optimizer=optimizer_Adam,
                        loss='categorical_crossentropy', metrics=['accuracy'])

# training 过程中的 自动调节函数
# Reduce LR On Plateau = 减少学习率，当某一个参数达到一个平台期 自动的 把上面优化器中的 lr 减小

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=0.00001)


# %%
# 模型训练
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          validation_data=(X_val, Y_val), callbacks=[learning_rate_reduction])


# %%
# 保存训练好的模型
model.save('D:\\Python_project\\Jupyter_project\\Three_week\\5.cnn_2\\cnn_history_model.h5')

# %%
# 重新加载训练好的模型
from keras.models import load_model
rnn_model = load_model('D:\\Python_project\\Jupyter_project\\Three_week\\5.cnn_2\\cnn_history_model.h5')
# %%

# 生成学习曲线 和损失函数 随着epoch的变化曲线
# 模型的学习效果怎么样？ 能找到适合的epoch吗？
# 简单的评价标准应该用什么？
# 尝试改变模型参数 生成不同的学习曲线 比较
# 提示 从epoch>优化器>损失函数>学习率>dropout有无 依次调试
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# %%
# 生成10个标签的混淆矩阵
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# %%
### 打印出认错的数字

errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 3
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 9 errors 
most_important_errors = sorted_dela_errors[-9:]

# Show the top 9 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

# %%
#optional 计算每个标签的tpr fpr
from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
y_score = model.predict(X_test)
for i in range(num_classes):
     fpr[i], tpr[i], _ = roc_curve(Y_test[:,i], y_score[:,i]) #
    # AUC Area Under the Curve
     roc_auc[i] = auc(fpr[i], tpr[i])
#y_pred_keras = model.predict(X_test).ravel()
##fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_pred_keras)
#y_pred_keras

# %%
#画出ROC
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

# %%
# 打印出模型图片 尝试自己改动模型 
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

# %%
