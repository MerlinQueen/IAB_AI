# %%
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input, VGG16
from keras import backend as K
import numpy as np

# %%
img_width, img_height = 224, 224
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
# 加载VGG16模型
input_tensor= Input(shape=input_shape)         
model = VGG16(include_top=False,weights='imagenet',input_tensor=input_tensor)
top_model = Sequential()
top_model.add(Flatten(input_shape=model.shape[1:]))
top_model.add(Dense(256,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1,activation='sigmoid'))
# 加载权重
new_model =Sequential()
for l in model.layers:
    new_model.add(l)
new_model.add(top_model)

# %%
# 设置不需要微调的layers的trinable属性
from keras.optimizers import SGD
for layers in new_model.layers[:4]:
    layers.trinable = False

# 重新编译
new_model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# %%
train_data_dir = r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\train'
validation_data_dir = r'D:\Python_project\Jupyter_project\Four_week\dogs_vs_cats_datas\validation'
nb_train_samples = 1083
nb_validation_samples = 400
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
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# %%模型训练
new_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# %%
# 模型验证
import cv2 as cv
img = cv.resize(cv.imread(r'D:\Python_project\Jupyter_project\Four_week\3.VlexNet_CV\dogs-vs-cats\test\1.jpg'), (img_width, img_height)).astype(np.float32)

x = img_to_array(img)
x = np.expand_dims(x,axis =0)

relust = model.predict(x)

if relust[0] <=0:
    print('猫')
else:
    print('狗')