#!/usr/bin/env python
# encoding: utf-8
import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.callbacks import TensorBoard
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

from keras.applications.imagenet_utils import decode_predictions


# 建立模型
model = Sequential()
model.add(Conv2D(32, 3, activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))
# 编译
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# 从图片中直接产生数据和标签
train_generator = train_datagen.flow_from_directory('I:/DeepFashion/tensorflow/keras_data/train',
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory('I:/DeepFashion/tensorflow/keras_data/test',
                                                        target_size=(150,150),
                                                        batch_size=32,
                                                        class_mode='categorical')
model.fit_generator(train_generator,
                    steps_per_epoch=1000,
                    epochs=2,
                    validation_data=validation_generator,
                    validation_steps=200)
# 保存整个模型
model.save('model.hdf5')

# 保存模型的权重
model.save_weights('model_weights.h5')

# 保存模型的结构
json_string = model.to_json()
open('model_to_json.json','w').write(json_string)
yaml_string = model.to_yaml()
open('model_to_yaml.yaml','w').write(json_string)

# 测试自己的图片
file_path = 'I:/DeepFashion/tensorflow/keras_data/img_00000023.jpg'
img = image.load_img(file_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
y = model.predict(x)
Fine_Pred = [result.argmax() for result in y][0]
print(y)
count = 0
for i in y[0]:
    percent = '%.2f%%' % (i * 100)
    print(count+1, '', percent)
    count += 1


