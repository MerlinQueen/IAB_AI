# %%
# 导入模块
import numpy as np 
import pandas as pd # 数据处理分析包
import seaborn as sns # 基于matplotlib的一个数据可视化库
from matplotlib import pyplot as plt 
# 线性模式中的逻辑回归模型
from sklearn.linear_model import  LogisticRegression
# 模型选择中的 训练和测试数据集分割
from sklearn.model_selection import train_test_split

#%%
# 1.读数据
# pd.DataFrame(data)   建立excel表形式的数据表
# df = pd.DataFrame(pd.read_csv("D:\\Python_project\\Jupyter_project\\Two_week\\neural_network_practice_2\\1_data\\heart.csv"))
# 用pandas 读取 csv文件
df = pd.read_csv("D:\\Python_project\\Jupyter_project\\Two_week\\neural_network_practice_2\\1_data\\heart.csv")
# 显示前5行数据
df.head(5)

#%%
# 2.查看target中(样本)值的个数是否是1:1
df['target'].value_counts()

#%%
# 用统计图countplot显示target列中的样本数量
# 使用统计图显示每个分类箱中的观测则,palette:调色板,bwr,蓝白红
sns.countplot(x='target',data=df,palette='bwr')
plt.show()

#%%
# 统计没有患心脏病的和患有心脏病的百分比
countNoDisease = len(df[df['target'] == 0])
countHaveDisease = len(df[df['target'] == 1])
print("不患病的百分比:{:.2f}".format((countNoDisease/(len(df['target']))*100)))
print("患病者的百分比:{:.2f}".format((countHaveDisease/(len(df['target']))*100)))



#%%
# 用统计图表示男女人数
sns.countplot(x='sex',data=df,palette='mako_r')
# 给x轴添加标签
plt.xlabel('Sex(0:famele , 1= male)')
plt.show()

#%%
# 计算男女数量的百分比
countFemale = len(df[df['target']==0])
countMale = len(df[df['target']==1])
print("女性患者占比:{:.2f}".format((countFemale/(len(df['target']))*100)))
print("男性患者占比:{:.2f}".format((countMale/(len(df['target']))*100)))

#%%
# grounby 分组
# 分组查看均值
df.groupby('target').mean()

#%%
# 交叉表crosstab 行列都有分组的表
# A交叉表示表示两个或者多个变量之间关系的表
# 查看年龄与患病人数之间的关系
pd.crosstab(df['age'],df['target']).plot(kind="bar",figsize=(20,6))
plt.title('心脏病的频率与年龄之间的关系')
plt.xlabel('Age')
plt.ylabel('频率')
# 以图片的形式存储
plt.savefig('heartDiseaseAndAges.png')
plt.show()
#%%
pd.crosstab(df['sex'],df['target']).plot(kind = 'bar',figsize = (15,6),color=['#1CA53B','#AA1111'])
plt.title("心脏病的患病频率与性别之间的关系")
plt.xlabel("sex(0:女性,1:男性)")
# 获取或设置x轴的当前标记位置和标签
plt.xticks(rotation= 0)
# legend图例,图例说明
plt.legend(["患心脏病","不患心脏病"])
plt.ylabel("频率")
plt.show()

#%%
# scatter散布,散点分布图
plt.scatter(x=df.age[df['target']==1],y=df.thalach[df['target']==1],c='red')
plt.scatter(x=df.age[df['target']==0],y=df.thalach[df['target']==0])
plt.legend(["Disease","Not Disease"])
plt.xlabel("age")
plt.ylabel("MAximun heart rate") # 最大心脏率
plt.show()

#%%
pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()

#%%
pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()

#%%
# dummy:虚拟的   prefix 前缀
# 把分类变量转成虚拟变量即one-hot 独热值编码
# 因为Cp,thal,slope是虚拟变量,我们把它转成虚拟变量
a = pd.get_dummies(df['cp'],prefix = "cp")
b = pd.get_dummies(df['thal'],prefix='thal')
c = pd.get_dummies(df['slope'],prefix = 'slope')

#%%
# concat合并多个数组
# 把独热值编码后的数组合并到df数据表中
df = pd.concat([df,a,b,c],axis = 1)
df.head(5)


# %%
# drop() 选择性的删除某一行或者列
# columns 列
df = df.drop(columns = ['cp','thal','slope'])
df.head(5)

#%%
# 创建逻辑回归模型
# 我们可以使用sklearn函数或者自己写的函数来创建模型
# 首先我们要写自己的函数,然后用sklearn函数计算模型的准确率等得分
# df.target.values 查看某一列的所有值
y = df['target'].values
# 把含有输出结果'target'的列的值删除,然后把每一列的值赋给x_data
x_data = df.drop(['target'],axis = 1)
df.head()
#%%
# 获取数据归一化处理后所有列的值
# 按照最大值-最小值归一化使其在(0-1)之间 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
# 进行数据拆分,把80%的数据当训练数据,20%的作为测试数据
# x_train 训练得输入值,y_train 对应的输出值 y = f(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


#%%
# 用Sklearn 的逻辑回归算法训练模型并计算准确率

#  acucuracies 准确率
# 创建一个精确度的字典,存放不同训练方式下的准确度
accuracies = {}

lr = LogisticRegression()
# 根据给定的训练数据拟合模型,逻辑回归算法拟合fit 拟合
lr.fit(x_train,y_train)
# 再用测试数据集测测试模型的准确度得分
acc = lr.score(x_test,y_test)*100
# 把对应的键值修改
accuracies['Logistic Regression'] = acc
print("Test Accuracy{:.2f}%".format(acc))


#%%
# K-Nearest Neighbour (KNN) Classification
# 用K最邻近算法分类
# 从邻近算法包中导入KNN分类函数
from sklearn.neighbors import KNeighborsClassifier
# 邻近值n_neighbors K值设置为
knn  = KNeighborsClassifier(n_neighbors= 2)
# 用KNN算法拟合
knn.fit(x_train,y_train)
# 用测试数据预测,并显示结果
prediction=knn.predict(x_test)
print("{}NN 得分:{:.2f}%".format(2,knn.score(x_test,y_test)*100))


#%%
# 用图表的方法尝试找最佳K值
scoreList = []
# K n_neighbors的值从1-19
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    scoreList.append(knn2.score(x_test,y_test))


plt.plot(range(1,20),scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel('k value')
plt.ylabel('scorce')
plt.show()

acc = max(scoreList)*100
accuracies["KNN"] = acc
print("MAximun KNN Score is{:.2f}%".format(acc))
# 从图像上可以看出 看k =  3,7,8时结果最好
#%%
# 支持向量机SVM算法
from sklearn.svm import SVC

#%%
svm = SVC(random_state=1)
svm.fit(x_train,y_train)
acc = svm.score(x_test,y_test)*100
accuracies['SVM'] = acc
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))

#%%
# Naive Bayes Algorithm
# 朴素贝叶斯算法 
from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()
nb.fit(x_train,y_train)
acc = nb.score(x_test,y_test)*100
accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))


#%%
# 决策树算法
# Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
acc = dtc.score(x_test,y_test)*100
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))


#%%
# 随机森林分类法
# ensemble :全体,总的
# forest:森林
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000,random_state=1)
rf.fit(x_train,y_train)
acc = rf.score(x_test,y_test)*100
accuracies['Random Forest'] = acc
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))



#%%
# 比较各个模型的准确率
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
# y轴以10为单位
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
# 画出条形图 bar 棒,条
sns.barplot(x=list(accuracies.keys()),y=list(accuracies.values()),palette=colors)
plt.show()

#%%
