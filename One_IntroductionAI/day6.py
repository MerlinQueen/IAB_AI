# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
# Project1:对文件diabetes.csv中的数据进行处理，将文件夹pima-diabetes放在E：盘的input文件夹下

#%%
# from pandas import *
import pandas as pd
import numpy as np
# seaborn 数据可视化工具包
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
from scipy.stats import mode
import os
print(os.listdir("D:/Python_project/Jupyter_project/One_week/neural_network_practice_1"))


#%%
################################读入diabetes.csv文件到diabetes################################### 
# 利用pandas读取文件 pd.read_csv("path")  Dataframe生成数据表
diabetes = pd.DataFrame(pd.read_csv("D:/Python_project/Jupyter_project/One_week/neural_network_practice_1/pima-diabetes/diabetes.csv"))
################################读入diabetes.csv文件到diabetes###################################
diabetes

#%%
################################显示diabetes的前5行################################### 
diabetes.head(5)
################################显示diabetes的前5行################################### 
# %%
# 显示disbetes的后5行
diabetes.tail(5)

#%% [markdown]
# Check for Summary Statistics

#%%
################################显示diabetes的详细信息，包括样本数、均值、方差等等################################## 
diabetes.describe()

################################显示diabetes的详细信息，包括样本数、均值、方差等等################################## 


#%%
################################显示diabetes中Outcome列中每种值出现的次数################################## 
diabetes['Outcome'].value_counts()

################################显示diabetes中Outcome列中每种值出现的次数################################## 

#%% [markdown]
# ### Check missing values

#%%
# df = diabetes.head()
# df
################################显示diabetes中所有列的标题名################################## 
# 索引columns纵列标题名称
diabetes.columns

################################显示diabetes中所有列的标题名################################## 


#%%
################################显示diabetes中所有列的值里面为0的个数################################## 
# 方法一取出所有列中==0的值进行统计
print((diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']] == 0).sum())
# 直接取出标题索引中 ==0 的值进行统计

################################显示diabetes中所有列的值里面为0的个数################################## 


#%%
################################将diabetes中所有列的值中为0的值替换为NaN################################## 
# loc[n:m,"str"] 提取从第n行到第m行,标题head为"str"的值
diabetes.loc[:,['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]   = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0,np.NaN)
################################将diabetes中所有列的值中为0的值替换为NaN################################## 
diabetes.head()


#%%
################################统计diabetes中所有列的值中为null的个数################################## 
# df.isnull() 判断空值
(diabetes.isnull()).sum()
################################统计diabetes中所有列的值中为null的个数################################## 

#%% [markdown]
# ## Dealing with missing values
#%% [markdown]
# ## A. Drop rows having NaN

#%%
print("Size before dropping NaN rows",diabetes.shape,"\n")

################################将diabetes中包含NaN的行删掉################################## 
# df.dropna() 清除包含缺失值NaN的行
nan_dropped = diabetes.dropna()
################################将diabetes中包含NaN的行删掉################################## 
print(nan_dropped.isnull().sum())
print("\nSize after dropping NaN rows",nan_dropped.shape)
#%% [markdown]
# ### **Project2、数据分析的一系列问题**

#%%
vector = np.random.chisquare(1,500)
print(vector)
################################打印vector的均值################################## 
print("均值means\n",np.mean(vector))

################################打印vector的均值################################## 
# 打印标准值
print("\nSD",np.std(vector))
# 打印最大最小值
print("\nRange\n",max(vector)-min(vector))

#%% [markdown]
# ### Reshape

#%%
vector.shape


#%%
################################将一维向量vector转成大小为500*1的二维向量并赋值给row_vector################################## 

# reshape(m,-1) #改变维度为m行、1列
# reshape(-1,m) #改变维度为1行、m列
# reshape(1,-1)转化成1行：
# reshape(-1,1)转换成1列：
# reshape(2,-1)转换成两行：
row_vector = vector.reshape(1,-1)
################################将一维向量vector转成大小为500*1的二维向量并赋值给row_vector################################## 
row_vector.shape
print(row_vector)


#%%
col_vector = vector.reshape(1,-1)
col_vector.shape


#%%
################################将一维向量vector转成大小为10*50的二维向量并赋值给maxtrix################################## 
matrix = vector.reshape(10,50)
################################将一维向量vector转成大小为10*50的二维向量并赋值给maxtrix################################## 
matrix.shape
print(matrix)

#%% [markdown]
# ### Pivot Table

#%%
#Determine pivot table
# 以字典键和列表键值对的形式存储数据
df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                       "bar", "bar", "bar", "bar"],
                    "B": ["one", "one", "one", "two", "two",
                          "one", "one", "two", "two"],
                    "C": ["small", "large", "large", "small",
                          "small", "large", "small", "small",
                          "large"],
                    "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                    "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})


#%%
df.head(9)


#%%
###################################想办法生成以下表格，行为A、B，列为C###################################
# pd.pivot_table 生成数据透视表
table = pd.pivot_table(df,values='D',index=['A','B'],columns='C',aggfunc=np.sum)

###################################想办法生成以下表格，行为A、B，列为C###################################
table

#%% [markdown]
# ### Merging dataframes

#%%
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                       index=[0, 1, 2, 3])
    

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                       index=[4, 5, 6, 7])
    

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                       'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                       index=[8, 9, 10, 11])
    
###################################将df1,df2,df3按行拼接到一起，并赋给result###################################
# pd.concat([数组1,数组2,数组3])  合并多个数组
result = pd.concat([df1,df2,df3])

###################################将df1,df2,df3按行拼接到一起，并赋给result####################################

print(result)


#%%






#%%
