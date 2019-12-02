# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#%%
# series对象,用的是列表
s = pd.Series([1,4,'ww','tt'])
s

#%%
sd = pd.Series({'Python':400,'c++':22,'c#':34})
sd 


#%%
# 自定义索引
s2 = pd.Series(['wang','man',24],index=['name','sex','age'])
s2['age']

#%%
# DataFrame是一种二维的数据结构
# Periods周期,要生成的个数,data_range 生成日期

dates  = pd.date_range('20180520',periods=6)
dates

#%%
# index:自定义索引
f3 = pd.DataFrame(data=None,columns=['name','marks','price'],index=['a','b','c'])
f3


#%%
# 查看dataframe顶部和尾部的数据
df = pd.DataFrame(pd.read_csv("D:\\Python_project\\Jupyter_project\\Two_week\\neural_network_practice_2\\1_data\\bike-sharing-dataset\\day.csv"))
df.head()
df.tail()
# 显示数据统计摘要,均值,最大值,标准值等
df.describe()
#%%
# 绘图
np.random.seed(100)
ts = pd.Series(np.random.randn(1000),index=pd.date_range("18/5/2018",periods=1000))
# 联合求和
ts = ts.cumsum()
ts.plot()

#%%
np.random.seed(1)
df = pd.DataFrame(np.random.randn(1000,4),index=ts.index,columns=['a','b','c','d'])

df = df.cumsum()
# 创建一个画布
plt.figure()
df.plot()
# 显示图例,a,b,c,d
plt.legend(loc='name')
#%%
# 数据的输入/输出
df = pd.DataFrame(pd.read_csv("D:\\Python_project\\Jupyter_project\\Two_week\\neural_network_practice_2\\1_data\\bike-sharing-dataset\\day.csv"))
# 写入到excel文件中
df.to_excel('bike.xlsx',sheet_name = 'Sheet1')
# 写入到csv文件
df.to_csv('bike_test.csv')
# 从Excel读取文件
# na_value = ["NA"]  空值赋值
pd.read_excel('bike.xlsx','Sheet1',index_col=None,na_values=['NA'])

#%%
# subplot把多个图画到一个平面上,陪衬
fig = plt.figure()
# add_subplot(n,m,5)画N行m列,画在第5副图上
ax = fig.add_subplot(2,2,1)
ax.set(xlim=[0.5,4.5],ylim=[-2,8],title='axes',ylabel='y',xlabel='x ')
plt.show()  


#%%
fig = plt.figure()
# 画在两行两列的第一个图上
ax1 = fig.add_subplot(3,2,1)
ax2 = fig.add_subplot(3,2,3)
ax3 = fig.add_subplot(3,2,6)
plt.show()

#%%
# 基本绘图
x = np.linspace(0,np.pi)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x,y_cos)
plt.show()

#%%
# 画散点图
# scatter:散点

x = np.arange(0,10)
y = np.random.randn(10)
plt.scatter(x,y,color = 'blue',marker='*')
plt.show()


#%%
# 画柿饼图
labels = 'frogs','hogs','dogs','logs'
sizes = [15,30,45,10]
explode = (0,0.1,0,0)
fig1,(ax1,ax2) = plt.subplots(2)
ax1.pie(sizes,labels = labels,autopct = '%1.1f%%',shadow = True)   
ax1.axis('equal')
ax2.pie(sizes,labels = labels,autopct = '%1.1f%%',shadow = True)   
ax2.axis('equal')
ax2.legend(labels=labels,locals = 'uppper right')
# %%
# 画三维图
from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
#三维线的数据
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# 三维散点的数据
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')


#%%
# 三维登高线图
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))
x = np.linspace(-6,6,30)
y = np.linspace(-6,6,30)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
ax.view_init(60, 35)

#%%
#  画箱型图
data = np.random.rand(10)
data2 = np.arange(10)
fig ,(ax1,ax2 ) =plt.subplots(2)
# boxplot箱型图
ax1.boxplot(data)
ax2.boxplot(data2)
plt.show()


#%%
# 画泡泡图
N = 10
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = 30 
area = (30*np.random.rand(N))*20

plt.scatter(x,y,s= area,c=colors,alpha=0.5)
plt.show()

#%%
