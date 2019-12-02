# %%
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,Concatenate,Activation
from keras.layers import AveragePooling2D,Input,BatchNormalization
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input
from keras import backend as K 
import numpy as np
from keras.models import Model
from keras.models import load_model


# %%
# 加载模型,进行预测
model = load_model(r'D:\Python_project\Jupyter_project\Five_week\1.Inception\Inception_model.h5')
# %%
import cv2 as cv
img_width,img_height = 50,50
if K.image_data_format()=='channels_first':
    input_shape = (3,img_width,img_height)
    bn_axis =1
else:
    input_shape = (img_width,img_height,3)
    bn_axis =3

img = cv.resize(cv.imread(r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\test\2.jpg'), (img_width, img_height)).astype(np.float32)

x = img_to_array(img)
x = np.expand_dims(x,axis =0)

score = model.predict(x)


print(score)

