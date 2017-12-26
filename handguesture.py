#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 23:56:41 2017

@author: purna
"""
from os import listdir
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,MaxPooling2D,Dropout,Conv2D,Flatten
from keras.preprocessing import image
from keras import backend as K
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input

def preprocessing():
    A_items = listdir('A/')
    B_items = listdir('B/')
    C_items = listdir('C/')
    processed_A_items = []
    processed_B_items = []
    processed_C_items = []
    for item in A_items:
        processed_A_items.append('A/'+ item)
    for item in B_items:
        processed_B_items.append('B/'+ item)
    for item in C_items:
        processed_C_items.append('C/'+ item)
    
    return processed_A_items,processed_B_items,processed_C_items

def image_to_matrix(path_A,path_B,path_C):
    tensor = []
    result = []
    try:
        tensor = np.load('tensor_data_modifie.npy')
        result = np.load('result.npy')
    except:
        for item in path_A:
             img = image.load_img(item, target_size=(64,64))
             y = image.img_to_array(img)
             y = np.expand_dims(y, axis=0)
             tensor.append(preprocess_input(y[0]))
             result.append([1,0,0])
             #print((tensor[0].shape))
        for item in path_B:
            img = image.load_img(item, target_size=(64,64))
            y = image.img_to_array(img)
            y = np.expand_dims(y, axis=0)
            tensor.append(preprocess_input(y[0]))
            result.append([0,1,0])
            #print((tensor[0].shape))
        for item in path_B:
            img = image.load_img(item, target_size=(64,64))
            y = image.img_to_array(img)
            y = np.expand_dims(y, axis=0)
            tensor.append(preprocess_input(y[0]))
            result.append([0,0,1])
            #print((tensor[0].shape))
        np.save('result',result)
        np.save('tensor_data_modifie',tensor)
        
    return tensor,result

def model_creation():
        model = Sequential()
        model.add(Conv2D(input_shape = [64,64,3],filters = 32,kernel_size = [5,5],padding = 'same',activation = 'relu'))
        model.add(MaxPooling2D(pool_size = [2,2]))
        model.add(Conv2D(filters = 64,kernel_size = [5,5],padding = 'same',activation = 'relu'))
        model.add(MaxPooling2D(pool_size = [2,2]))
        model.add(Flatten())
        model.add(Dense(256,activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3,activation='softmax'))
        model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
        return model

def model_call(model,image_matrices,result_values):
    try:
        model = load_model('hand_guesture_try2.h5')
    except:
        history = model.fit(image_matrices,result_values,epochs = 20,batch_size = 512,verbose = 1,validation_split = 0.1,shuffle = True)
        model.save('hand_guesture_try2.h5')
    return model
            

if __name__ == "__main__":
    path_A,path_B,path_C = preprocessing()
    print(len(path_A),len(path_B),len(path_C))
    image_matrices,result_values= image_to_matrix(path_A,path_B,path_C)
    print(image_matrices.shape)
    model = model_creation()
    print(model.summary())
    model_trained = model_call(model,image_matrices,result_values)
    #model.fit(image_matrices,result_values,epochs = 20,batch_size = 512,verbose = 1,validation_split = 0.3)
    matric = []
    item = '6.jpg'
    img = image.load_img(item, target_size=(64,64))
    y = image.img_to_array(img)
    y = np.expand_dims(y, axis=0)
    matric.append(preprocess_input(y[0]))
    matric = np.asarray(matric)
    X = model_trained.predict(matric)
    print((X[0][0]))
    for i in range (2):
        if int(X[0][i]) == 1:
            print('belongs to class',i)