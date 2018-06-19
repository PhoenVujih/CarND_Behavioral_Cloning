import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import pandas as pd
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from functions import *

# Model Architechture inspired by Nvidia
def nvidia_model(strides=(2,2), dropout=0.5):
    model = Sequential()
    # Preprocess the image
    model.add(Lambda(lambda x:x/127.5-1., input_shape=(img_h,img_w,img_d)))
    # Conv NN
    model.add(Conv2D(24,(5,5), strides=strides, activation='elu'))
    model.add(Conv2D(36,(5,5), strides=strides, activation='elu'))
    model.add(Conv2D(48,(5,5), strides=strides, activation='elu'))
    model.add(Conv2D(64,(3,3), activation='elu'))
    model.add(Conv2D(64,(3,3), activation='elu'))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(200, activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='elu'))    
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='mse')
    return model

# Read the data list. The csv here was recombined with normal and reversal loops
samples = pd.read_csv('data_new.csv')
# Split the dataset in to train and valid sets.
samples_train, samples_valid = train_test_split(samples, test_size=0.2, random_state=0)

model = nvidia_model()
# model.load_weights('model-037.h5')  #load the existed weights for modification
checkpoint = ModelCheckpoint('model{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=1, mode='auto')

model.fit_generator(batch_generator(samples_train, True, batch_size = 80),
                    len(samples_train),
                    50,
                    max_queue_size=1,
                    validation_steps=len(samples_valid),
                    callbacks=[checkpoint],
                    validation_data=batch_generator(samples_valid, False),
                    verbose=1)

