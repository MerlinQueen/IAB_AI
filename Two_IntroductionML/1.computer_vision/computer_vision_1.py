# %%
# 导入图像
# 图像降维度
import math
import numpy as np 
import matplotlib.pyplot as plt 
from skimage import data,color,io
from skimage.transform import rescale,resize,downscale_local_mean
# 图像变换的函数,反射,缩放,旋转



#%%
# 练习一
# 转成灰度级图像
camera = data.camera()
img_camera = data.camera()
io.imshow(img_camera )
# 图像切片
camera[:10] = 0
# 把像素值小于87的遮起来,设置阈值生成掩膜
mask = camera < 87
# 把遮罩部分像素值改成白色
camera[mask] = 255

# 把图像的尺寸转成数组,设置生成掩模
inds_x = np.arange(len(camera))
# print("inds_X",inds_x)

inds_y = (4 * inds_x) % len(camera)
# print("inds_y",inds_y)

camera[inds_x, inds_y] = 0
io.imshow(camera)
# 获取图像的宽和高,也就是最右下角的那个点
l_x, l_y = camera.shape[0], camera.shape[1]
# 获取左上角的点
X, Y = np.ogrid[:l_x, :l_y]
outer_disk_mask = (X - l_x / 2)**2 + (Y - l_y / 2)**2 > (l_x / 2)**2
camera[outer_disk_mask] = 0

plt.figure(figsize=(4, 4))
plt.imshow(camera, cmap='gray')
plt.axis('off')
plt.show()

#%%
# 练习二
# anti:反的
# roate:选择
# rescale:图片缩放
# resize:改变尺寸
# Downscale:以整数因子对n维图像就那些下采样
# 把data自带的宇航员图片转成灰度图
img = color.rgb2gray(data.astronaut())
io.imshow(img)
#用rescale函数缩放图片,大于1放大,小于1缩小
# anti aliasing:反混叠
img_big = rescale(img,scale=2,anti_aliasing=False)
io.imshow(img_big)
# 用resize 改变图片的尺寸
img_resize = resize(img,(100,100))
io.imshow(img_resize)
# // 整除:只保留整数位, / 除法得到的是一个浮点数
# img.shape[0]图片的宽,img.shape[1]图片的高,img.shape[2]图片通道
img_resize_1 = resize(img,(img.shape[0]//4,img.shape[1]//4),anti_aliasing=True)
io.imshow(img_resize_1)
# 用downscale_local_mean函数缩小宇航员图片
# downscale_local_mean通过局部平均值对N维图像进行下采样
# factors: 因子(4,3) 缩放成宽:高 = 4:3的图像
img_dscale = downscale_local_mean(img,(5,2))
io.imshow(img_dscale)
# plt 部分
# plt.subplots返回一个n行n列的子图,把几幅图放在同一个坐标上显示
fig, axes = plt.subplots(nrows=2, ncols=2)

ax = axes.ravel()
# axes,axes[0]便是第一个子图
ax[0].imshow(img, cmap='gray')
ax[0].set_title("Original image")

ax[1].imshow(img_big, cmap='gray')
ax[1].set_title("Rescaled image (aliasing)")

ax[2].imshow(img_resize, cmap='gray')
ax[2].set_title("Resized image (no aliasing)")

ax[3].imshow(img_dscale, cmap='gray')
ax[3].set_title("Downscaled image (no aliasing)")

ax[0].set_xlim(0, 512)
ax[0].set_ylim(512, 0)
plt.tight_layout()
plt.show() 

#%%
# 练习三
# 学习使用几何变换函数
from skimage import transform as tfm
#%%
# 相似变换:形状不变,方向和大小改变:相似性,反射,旋转等
# 生成一个Kernel矩阵,作用:进行点积作相似变换
# 创建一个可以进行相似变换的Kernel 矩阵
# scale:缩放比例
# rotation:旋转角度
# translation:反射变换
tform_kernel_big= tfm.SimilarityTransform(scale=2)
# params:参数   打印出内核元素
print(tform_kernel_big.params,"\n")
# 旋转图像的内核
tform_kernel_small= tfm.SimilarityTransform(rotation=math.pi/2)
print(tform_kernel_small.params,"\n")
# 反射的内核
tform_kernel_reflect= tfm.SimilarityTransform(translation=(0,1))
print(tform_kernel_reflect.params,"\n")
# 综合内核
tform_kernel = tfm.SimilarityTransform(scale=2,rotation=90,translation=(0,1))
print(tform_kernel.params,"\n")
# 调用transform.warp()将生成kernel矩阵作用在text图像上
# 复制内核矩阵
# 获取内置图片
text = data.text()
# warp():弯曲,变形
tfm_kernel = tfm.SimilarityTransform(scale=1,rotation=math.pi/4,translation=(text.shape[0]/2,-100))
rotated = tfm.warp(text,tfm_kernel)
back_rotated = tfm.warp(rotated,tfm_kernel.inverse)
fig,ax = plt.subplots(nrows=3)

ax[0].imshow(text, cmap="gray")
ax[1].imshow(rotated, cmap="gray")
ax[2].imshow(back_rotated, cmap=plt.cm.gray)
for a in ax:
    a.axis('off')
plt.tight_layout()
#%%
# 练习四:均值滤波
# Mean filters
# local mean: 局部均值:计算平均灰度级
# percentile mean:百分数均值:仅使用百分数p0,p1之间的值
# bilateral mean:双边均值:g-s0和g + s1内部具有灰度级的结构元素的像素
from skimage.morphology import disk
from skimage.filters import rank 
img_2 = data.coins()
# disk(n)  以图像某个像素点为中心,选择周围n*n n行n列的像素做运算
selem = disk(3)
# 采用百分数均值滤波方式
percent_img = rank.mean_percentile(img_2,selem=selem,p0=0.1,p1=0.9)
# 采用双边均值滤波方式
bilateral_img = rank.mean_bilateral(img_2,selem=selem,s0=500,s1=500)
# 采用直接复制周围点滤波方式
normal_img = rank.mean(img_2,selem=selem)
# 显示部分
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                         sharex=True, sharey=True)
ax = axes.ravel()

titles = ['Original', 'Percentile mean', 'Bilateral mean', 'Local mean']
imgs = [img_2, percent_img, bilateral_img , normal_img]
for n in range(0, len(imgs)):
    ax[n].imshow(imgs[n], cmap=plt.cm.gray)
    ax[n].set_title(titles[n])
    ax[n].axis('off')

plt.tight_layout()
plt.show()


#%%
