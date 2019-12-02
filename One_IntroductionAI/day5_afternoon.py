# %%
import numpy as np 
from matplotlib import pyplot as plt 
from PIL import Image

#%%
A = np.array([[7,2],[3,4],[5,3]])
# 用奇异分解SVD方法求解矩阵A计算U,V,D
u,s,vh = np.linalg.svd(A)
print("u\n",u)
print("s\n",s)
print("vh\n",vh)

#%%
img = Image.open("D:/Python_project/pycharm-project/opencv/picture/pyramid.png")
print(img)
# LA转成灰度值,1:1位像素,RGB色彩图,CMYK图
imggray = img.convert("LA")
# 转成数组
imgmat = np.array(list(imggray.getdata(band= 0)),float)
# 转成矩阵
newimg = imgmat.reshape(512,512)
# 降维度获取u,s,vh 
u,s,vh = np.linalg.svd(newimg)
# newImg = imgmat.shape(imggray.size[1],imggray.size[0])
for i in [5,10,15,20,30,50]:
    # 重构
    reconstImg = np.matrix(u[:,:i])*np.diag(s[:i])*np.matrix(vh[:i,:])
    plt.imshow(reconstImg)
    plt.title("duck")
    plt.show()
# 创建一个画布显示出来
# plt.figure(figsize=(9,6))
# plt.imshow(newimg)
# plt.imshow(imggray)
# plt.show()
#%%
