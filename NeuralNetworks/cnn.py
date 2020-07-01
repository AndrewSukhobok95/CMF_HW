from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train = train_datagen.flow_from_directory('data/dog_cat/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test = test_datagen.flow_from_directory('data/dog_cat/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


def cnnModel():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (3, 3)))

    model.add(Flatten())

    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

cnn = cnnModel()
cnn.fit_generator(train,
                  steps_per_epoch = 8000,
                  epochs = 10,
                  validation_data = test,
                  validation_steps = 2000)


test_image_1 = image.load_img('data/dog_cat/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image_1 = image.img_to_array(test_image_1)
test_image_1 = np.expand_dims(test_image_1, axis = 0)

test_image_2 = image.load_img('data/dog_cat/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image_2 = image.img_to_array(test_image_2)
test_image_2 = np.expand_dims(test_image_2, axis = 0)

result = cnn.predict(test_image_1), cnn.predict(test_image_2)

print(result, train.class_indices)