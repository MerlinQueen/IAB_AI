# %%
# 加载模块
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,Concatenate,Activation
from keras.layers import AveragePooling2D,Input,BatchNormalization
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input
from keras import backend as K 
import numpy as np
from keras.models import Model
# %%
# 定义Inception网络结构
img_width,img_height = 50,50
if K.image_data_format()=='channels_first':
    input_shape = (3,img_width,img_height)
    bn_axis =1
else:
    input_shape = (img_width,img_height,3)
    bn_axis =3

x = Input(shape=input_shape)


# branch 分支1
branch1_out= Conv2D(16,(1,1),padding='same',use_bias = False)(x)
branch1_out = BatchNormalization(axis= bn_axis)(branch1_out)
branch1_out = Activation(activation='relu')(branch1_out)

# branch2 分支2
branch2_out = Conv2D(16,(1,1),padding='same',use_bias = False)(x)
branch2_out = BatchNormalization(axis =bn_axis)(branch2_out)
branch2_out = Activation(activation='relu')(branch2_out)
branch2_out = Conv2D(48,(3,3),padding ='same',use_bias = False)(branch2_out)
branch2_out = BatchNormalization(axis=bn_axis)(branch2_out)
branch2_out = Activation('relu')(branch2_out) 

# brach3 分支3
branch3_out = Conv2D(16, (1, 1),padding="same",use_bias=False)(x)   # use_bias不加偏置
branch3_out = BatchNormalization(axis=bn_axis)(branch3_out)     # axis_bn_axis  keras不同后端对应的axis不一样
branch3_out = Activation('relu')(branch3_out)
branch3_out = Conv2D(24, (5, 5),padding="same",use_bias=False)(branch3_out)
branch3_out = BatchNormalization(axis=bn_axis)(branch3_out)
branch3_out = Activation('relu')(branch3_out) 

# branch4 分支4
branch4_out = AveragePooling2D(pool_size=(3, 3),strides=(1, 1), padding='same', data_format=K.image_data_format())(x)
branch4_out = Conv2D(16, (1, 1),padding="same",use_bias=False)(branch4_out)
branch4_out = BatchNormalization(axis=bn_axis)(branch4_out)
branch4_out = Activation('relu')(branch4_out) 

# 把前面各层全连接起来
out = Concatenate(axis=bn_axis)([branch1_out,branch2_out,branch3_out,branch4_out])  
out = Conv2D(16,(1,1),padding='same',use_bias =False)(out)

# 全连接层
out = Flatten()(out)
out = Dense(48,activation='relu')(out)

# 输出层
out = Dense(1,activation='sigmoid')(out)

model = Model(x,out)


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# %%
model.summary()
# %%
from keras.utils import plot_model
plot_model(model, to_file=r'D:\Python_project\Jupyter_project\Five_week\1.Inception\model.png', show_shapes=True)
# %%
# 定义ImageDataGenerator
train_data_dir = r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\train'
validation_data_dir = r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\validation'
nb_train_samples = 10800
nb_validation_samples = 4000
epochs = 1
batch_size = 20

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# %%
# 训练模型
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# %%
# 模型评估
import cv2 as cv
img = cv.resize(cv.imread(r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\test\1.jpg'), (img_width, img_height)).astype(np.float32)

x = img_to_array(img)
x = np.expand_dims(x,axis =0)

relust = model.predict(x)

if relust[0] <=0:
    print('猫')
else:
    print('狗')

# %%
