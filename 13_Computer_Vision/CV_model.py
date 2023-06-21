import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D as AvgPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50

def load_train(path):

    datagen = ImageDataGenerator(validation_split=.25, rescale=1/255)

    datagen_flow = datagen.flow_from_dataframe(
        dataframe = pd.read_csv(path + 'labels.csv'),
        directory = path + '/final_files',
        x_col='file_name',
        y_col='real_age',
        class_mode='raw',
        batch_size=16,
        seed=111,
        subset='training'
    )
    
    return datagen_flow

def load_test(path):

    datagen = ImageDataGenerator(validation_split=.25, rescale=1/255)

    datagen_flow = datagen.flow_from_dataframe(
        dataframe = pd.read_csv(path + 'labels.csv'),
        directory = path + '/final_files',
        x_col='file_name',
        y_col='real_age',
        class_mode='raw',
        batch_size=16,
        seed=111,
        subset='validation'
    )
    
    return datagen_flow

def create_model(input_shape):

    model = Sequential()
    optimizer = Adam(lr=1e-4) 

    backbone = ResNet50(input_shape=input_shape, 
            weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 
            include_top=False)

    #backbone.trainable = False

    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer=optimizer, loss='mse',
                  metrics=['mae'])

    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=30, steps_per_epoch=None, validation_steps=None):

    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model

#create_model((150, 150, 3)).summary()
