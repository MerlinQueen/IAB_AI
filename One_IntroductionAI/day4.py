# %%
import numpy as np
x = np.array([1, 2, 3, 4])
#  x是一个矢量
x


# %%
# 创建一个矩阵
A = np.array([[1, 2], [2, 3], [3, 4]])
# A 是一个矩阵
A
# T是什么意思  transpose
A_t = A.T

A_t


# %%
# 加法 additon
A = np.array([[1, 2], [2, 3], [3, 4]])
B = np.array([[2, 5], [7, 4], [4, 3]])
C = A+B
C


#%%
# 矢量与矩阵的乘法
A = np.array([[1,2],[3,4],[5,6]])
B = np.array([2,3])
# 直接分别相乘
C  = A*B
print(C,"c\n")
# 点积,矩阵与向量相乘
# 1*2+2*3 = 8
D = np.dot(A,B)
print(D,"d\n")
#%%
# 单位矩阵与逆矩阵
A = np.array([[3,0,2],[2,0,-2],[0,1,1]])
A
# np.linalg
# inverse 求逆
# linage  线性代数
A_inv = np.linalg.inv(A)
A_inv
A_bis = np.dot(A,A_inv)
A_bis

#%%
from matplotlib import pyplot as plt 
x = np.arange(-10,10)
y = 2*x
y1 = -x+3
# figure 创建一个画布
plt.figure()
plt.plot(x,y)
plt.plot(x,y1)
# 定义x,y轴的范围
plt.xlim(0,3)
plt.ylim(0,3)
#  画轴 draw axes
#  vline 垂直轴
plt.axvline(x=0,color = "grey")
#  hline 水平轴
plt.axhline(y=0,color = "red")
plt.show()

#%%
# diagonal 变成对角线关系
lambdas = np.diag([6,2,2])
lambdas

#%%
A = np.array([[5,1],[3,3]])
A 
np.linalg.eig(A)

#%%
