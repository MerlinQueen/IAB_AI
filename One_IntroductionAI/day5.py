# %%
import numpy as np 
from matplotlib import pyplot as plt 
from PIL import Image
A = np.array([[1,2],[6,4],[3,2]])
A 
np.linalg.norm(A)


#%%
# 求特征向量和特征值
A = np.array([[3,6],[5,2]])
A
n,M = np.linalg.eig(A)
print("n\n",n)
print("M\n",M)
#%%
# 奇异分解对图像进行降维处理
plt.style.use("classic")
img = Image.open("D:/Python_project/pycharm-project/opencv/picture/pyramid.png")
# 转成灰度图像
imggray = img.convert("LA")
# 转成nump数组
imgmat = np.array(list(imggray.getdata(band=0)),float)
# 重定义原始图像的尺寸
imgmat.shape = (imggray.size[1],imggray.size[0])
plt.figure(figsize=(9,6))
plt.imshow(imgmat,cmap="gray")
plt.show()
print(imgmat)

#%%
A = np.array([[0,1],[1,1],[2,1],[3,1],[3,1],[4,1]])
B = np.array([[2],[4],[0],[2],[5],[3]])
A_plus = np.linalg.pinv(A)
print(A_plus)
coefs = np.dot(A_plus,B)
print("coef\n",coefs)

#%%
A = np.array([[8,-5],[6,-3]])
A
n,M = np.linalg.eig(A)
print("n\n",n)
print("M\n",M)

#%%
np.random.seed(123)
x = 5* np.random.rand(100)
y = 2* x+1+np.random.randn(100)

x = x.reshape(100,1)
y = y.reshape(100,1)
X = np.hstack([x,y])
# 矩阵的尺寸
print(X.shape)
# 用点表示他的位置
plt.plot(X[:,0],X[:,1],'.')
plt.show()

#%%
