# %%
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import Sequential
import pandas as pd
import numpy as np 
import keras

# %%
# 创建一个二分类的神经网络
# 数据集准备
train = pd.read_csv("D:\\Python_project\\Jupyter_project\\Two_week\\4.neural_network_practice_BP_3\\1.data\\train.csv")
test = pd.read_csv("D:\\Python_project\\Jupyter_project\\Two_week\\4.neural_network_practice_BP_3\\1.data\\test.csv")
print(train.head(5))
print(test.head(5))
# 包含仅有0,1二进制的标签行
X = train[train['label'].isin([0,1])]

# 目标变量
Y = train[train['label'].isin([0,1])]['label']
#  去掉lable列的数据 axis = 1   列
X = X.drop(['label'],axis =1)
X.head(5)
# %%
# 创建激活函数
# 创建一个sigmoid激活函数
def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
    return s


# %%
# 创建神经网络结构,输入层,隐藏层,输出层
def network_architecture(X,Y):
    n_x = X.shape[0]
    n_h = 10
    n_y = Y.shape[0]
    return (n_x,n_h,n_y)



# %%
# 定义神经网络参数,权重,偏差,初始权重值
def define_network_parameters(n_x,n_h,n_y):
    W1 = np.random.rand(n_h,n_x)*0.01  # 随机初始化权重
    b1 = np.zeros((n_h,1))  # 初始化偏置设为0

    # 输出层参数初始化
    W2 =np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    return{"W1":W1,'b1':b1,'W2':W2,'b2':b2}
    



# %%
# 实现正向传播
def forward_propagation(X,params):
    Z1 = np.dot(params['W1'],X)+params['b1']
    A1 = sigmoid(Z1)

    # 计算Z2 和A2
    Z2 = np.dot(params['W2'],A1)+params['b2']
    A2 = sigmoid(Z2)

    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}  

# %%
# 计算误差函数的损失成本(期望值与实际值之差)
def compute_error(Predicted,Actual):  
    logprobs = np.multiply(np.log(Predicted),Actual)+np.multiply(np.log(1-Predicted),1-Actual)
    cost = -np.sum(logprobs)/Actual.shape[1]
    return np.squeeze(cost)

# %%
# 实现向后传播
def backward_propagation(params, activations, X, Y):                ##########反向传播算法，计算导数############
    m = X.shape[1]
    
    # output layer
    dZ2 = activations['A2'] - Y # compute the error derivative 
    dW2 = np.dot(dZ2, activations['A1'].T) / m # compute the weight derivative 
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m # compute the bias derivative
    
    # hidden layer
    dZ1 = np.dot(params['W2'].T, dZ2)*(1-np.power(activations['A1'], 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1,keepdims=True)/m
    
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
# %%
# 反向传播算法更新参数
def update_parameters(params, derivatives, alpha = 1.2):           ##########反向传播算法，更新参数############
    # alpha is the model's learning rate 
    
    params['W1'] = params['W1'] - alpha * derivatives['dW1']
    params['b1'] = params['b1'] - alpha * derivatives['db1']
    
    #  请计算W2和b2更新之后的值
    params['W2'] = params['W2'] - alpha * derivatives['dW2']
    params['b2'] = params['b2'] - alpha * derivatives['db2']
   
    
    return params
# %%
# 编译和训练模型
# 编译网络
def neural_network(X, Y, n_h, num_iterations=100):                      ############编译网络
    n_x = network_architecture(X, Y)[0]
    n_y = network_architecture(X, Y)[2]
    
    params = define_network_parameters(n_x, n_h, n_y)                  ############初始化参数  
    for i in range(0, num_iterations):
        results = forward_propagation(X, params)                       ############前向传播 
        error = compute_error(results['A2'], Y)
        derivatives = backward_propagation(params, results, X, Y)      ############反向传播
        params = update_parameters(params, derivatives)                ############更新参数
    return params
    

# %%
y = Y.values.reshape(1, Y.size)
x = X.T.as_matrix()
model = neural_network(x, y, n_h = 10, num_iterations = 10)        #生成一个新的网络


# %%
# 预测
def predict(parameters, X):
    results = forward_propagation(X, parameters)
    print (results['A2'][0])
    predictions = np.around(results['A2'])    
    return predictions

predictions = predict(model, x)
print ('Accuracy: %d' % float((np.dot(y,predictions.T) + np.dot(1-y,1-predictions.T))/float(y.size)*100) + '%')

# %%
