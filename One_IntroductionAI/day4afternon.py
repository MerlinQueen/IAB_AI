# %%
import numpy as np 
from matplotlib import pyplot as plt 


#%%
# 矩阵的直接相乘:
A = np.array([[5,1],[3,3]])
B = np.array([[1,2],[5,4]])
D = A*B
print(D,"D\n")
# 矩阵乘积,表示一个矩阵变换的合成
C = np.dot(A,B)
print(C,"c\n")

#%%
# 求出M的逆矩阵
M = np.array([[3,6],[5,2]])
print(np.linalg.inv(M))

#%%
t = np.linspace(0,2*np.pi,100)
x = np.cos(t)
y = np.sin(t)
plt.figure()
plt.plot(x,y)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.show()




#%%
# 随机抽取3行2列的(0,1)之间的样本
# print(np.random.rand(3,2),"\n")

# 随机生成服从正太分布的数组
# print(np.random.randn(3,3),"\n")

print(np.random.seed(10),"\n")

# print(np.eye(5))
#%%
