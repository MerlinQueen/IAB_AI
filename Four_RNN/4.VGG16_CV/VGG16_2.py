# %%
from keras.preprocessing.image import ImageDataGenerator , img_to_array 
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout,Add
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras import backend as K 
import numpy as np 
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D

# %%
# 定义网络结构
img_width,img_height = 224,224
if K.image_data_format() =='channels_first':
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)

# %%
# 方案一创建模型
def VGGNet_Conv():
    model = Sequential()
    # 第一块64
    #       1
    model.add(Conv2D(64,(3,3),input_shape=input_shape,activation='relu',padding='same'))
    #       2
    model.add(Conv2D(64,(3,3),activation ='relu',padding='same'))
    #    池化
    model.add(MaxPool2D((2,2),2))
    # 第二块128
    #       1
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    #       2
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    #       池化
    model.add(MaxPool2D((2,2),2))
    # 第三块区域256
    #       1
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    #       2
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    #       3
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    #       池化
    model.add(MaxPool2D((2,2),2))
    # 第四块区域512
    #       1
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))  
    #       2
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))  
    #       3
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    #       池化
    model.add(MaxPool2D((2,2),2))
    # 第五块区域 512
    #       1
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    #       2
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    #       3
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    #       池化
    model.add(MaxPool2D((2,2),2))
  
    return model

model = VGGNet_Conv()
model.summary()

# %%
def VGG_Dense():
    # 第六块区域全连接 4096
    model = Sequential()
    #       拉平
    model.add(Flatten())
    #       1
    model.add(Dense(4096,activation='relu'))
    #       最优失活参数0.5
    model.add(Dropout(0.5))
    #       2
    model.add(Dense(4096,activation='relu'))
    #       最优失活参数0.5
    model.add(Dropout(0.5))
    #       输出层1000,但是猫狗分类只有2
    model.add(Dense(1,activation='sigmoid'))
    
    return model
    

# %%
# 方案二创建模型
    
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))
model.summary()
# %%
# 模型编译
model.compile(optimizer='rmsprop',loss = 'binary_crossentropy',metrics=['accuracy'])
# %%
# 保存模型的结构
model_json = model.to_json()
with open(r'D:\Python_project\Jupyter_project\Four_week\4.VGG16_CV\vgg_model.json','w') as json_file:
    json_file.write(model_json)
    json_file.close()

# %%
# 加载保存的模型
from keras.models import model_from_json

with open(r'D:\Python_project\Jupyter_project\Four_week\4.VGG16_CV\vgg_model.json','r') as json_file:
    model_json = json_file.read()
    model_vgg = model_from_json(model_json)
    json_file.close()

model_vgg.summary()

# %%
# 加载预训练的权值 
import h5py
f = h5py.File(r'D:\Python_project\Jupyter_project\Four_week\4.VGG16_CV\vgg16_weights.h5')
for k in range(f.attrs['nb_layers']):
    if k >= len(model_vgg.layers) - 1:
        # we don't look at the last two layers in the savefile (fully-connected and activation)
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    layer = model_vgg.layers[k]

    if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

layer.set_weights(weights)

f.close()


# %%
# 定义加入新的layers
top_model =Sequential()
top_model.add(Flatten(input_shape=model_vgg.output_shape[1:]))  
top_model.add(Dense(256,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1,activation='sigmoid'))
# top_model.load_weights('models/bottleneck_40_epochs.h5')
model_vgg.add(top_model)

# %%
# 设置不需微调的layers的trainable属性
for layer in model_vgg.layers[:25]:
    layer.trainable = False
#compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model_vgg.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# %%
# 数据批量预处理
train_data_dir = r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\train'
validation_data_dir = r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\validation'
nb_train_samples = 1083
nb_validation_samples = 400
epochs  =1
batch_size = 20
train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255)

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
# 模型训练
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)