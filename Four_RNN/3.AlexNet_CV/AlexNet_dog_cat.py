# %%
# 导入必要的包
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input
from keras import backend as K 
import numpy as np 

# %%
# 构建模型
img_width,img_height =227,227
if K.image_data_format() == "channels_first":
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)

# 构建模型
img_width,img_height =227,227
if K.image_data_format() == "channels_first":
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)

model = Sequential()
model.add(Conv2D(96,(11,11),input_shape=input_shape,activation = 'relu',strides=4))
model.add(MaxPool2D((3,3),2))
model.add(Conv2D(256,(5,5),activation ='relu',padding='same'))
model.add(MaxPool2D((3,3),2))

model.add(Conv2D(384,(3,3),activation ='relu',padding='same'))


model.add(Conv2D(384,(3,3),activation ='relu',padding='same'))
model.add(Conv2D(256,(3,3),activation ='relu',padding='same'))
model.add(MaxPool2D((3,3),2))
#  卷积层的深度越大,所需要的参数越少
model.add(Flatten())

model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()
# %%
# 模型编译
model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics=['accuracy'])



# %%
# 导入数据批量处理
trian_data_dir = r'D:\Python_project\Jupyter_project\Four_week\3.VlexNet_CV\dogs-vs-cats\train'
validation_data_dir = r'D:\Python_project\Jupyter_project\Four_week\3.VlexNet_CV\dogs-vs-cats\validation'
# 样本分割
nb_train_samples =  10835
nb_validation_samples = 4000
# 设置批次和迭代次数
epochs = 1
batch_size = 100    
# 数据归一化处理
train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

# 
test_datagen= ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(trian_data_dir,target_size=(img_width,img_height),batch_size=batch_size,class_mode='binary')    

validation_generator = test_datagen.flow_from_directory(validation_data_dir,target_size=(img_width,img_height),batch_size=batch_size,class_mode='binary')   



# %%
# 模型训练
model.fit_generator(train_generator,steps_per_epoch=nb_train_samples//batch_size,epochs=epochs,validation_data=validation_generator,validation_steps=nb_validation_samples//batch_size)


# %%
# 保存模型
model.save("D:\\Python_project\\Jupyter_project\\Four_week\\3.VlexNet_CV\\AlexNet_dog_cats_model.h5")
# %%
import cv2 as cv
# %%
img = cv.resize(cv.imread(r'D:\Python_project\Jupyter_project\Four_week\3.VlexNet_CV\dogs-vs-cats\test\1.jpg'), (img_width, img_height)).astype(np.float32)

x = img_to_array(img)
x = np.expand_dims(x,axis =0)

relust = model.predict(x)

if relust[0] <=0:
    print('猫')
else:
    print('狗')
# %%
