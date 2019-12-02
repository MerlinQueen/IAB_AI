# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
# %%
# 创建一个长度为10的一维全为0的nda
from matplotlib import pyplot as plt
import numpy as np
a = np.zeros(10)
a[4] = 1
print(a, "\n")

# 第二题
a = np.arange(0, 15, 1).reshape(5, 3)
b = np.arange(0, 12, 2).reshape(3, 2)
print(a, "\n")
print(np.dot(a, b), "\n")
# %%
x = [1, 2, 3, 4, 5, 6, 8, 9]
x = np.asarray(x)
y = x**3+3
plt.plot(y, x)
plt.show()


# %%
a = np.array([[1,3,4],[12,3,5],[123,3,3]])
print(a.shape)
print(a.ndim)
print(a.itemsize)

#%%
x = np.array([[1,3,4,6],[12,3,5,6],[123,3,3,8]])
y = np.array([[[1,2,1,8],[1,8,2,1],[1,8,2,1]]])

print(x,x.shape)
# print(x+y)

#%%
a = np.random.randn(5, 3)  #返回m*n的多维数组
print(a)
#%%
m = np.transpose(x)
print(m,m.shape)



#%%
m = np.arange(0,10,1)
print(m)
# print(np.reciprocal(m))


#%%
x = np.array([[1,3,4,6],[12,3,5,7],[123,3,3,8]])
y = x[1,3]
print(y)

#%%
