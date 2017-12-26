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
from keras.layers import Dense, Activation,MaxPooling2D,Dropout,Conv2D,Flatten,Reshape
from keras.preprocessing import image
from keras import backend as K
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from matplotlib import pyplot as plt

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
        tensor = np.load('tensor_data_modified_try3.npy')
        result = np.load('result_try3.npy')
    except:
        for item in path_A:
             img = image.load_img(item, target_size=(64,64))
             img1 = np.asarray(img)
             lower_blue = np.array([3,50,50])
             upper_blue = np.array([33,255,255])
             hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
             mask = cv2.inRange(hsv, lower_blue, upper_blue)
             tensor.append(mask)
             result.append([1,0,0])
             #print((tensor[0].shape))
        for item in path_B:
            img = image.load_img(item, target_size=(64,64))
            img = image.load_img(item, target_size=(64,64))
            img1 = np.asarray(img)
            lower_blue = np.array([3,50,50])
            upper_blue = np.array([33,255,255])
            hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            tensor.append(mask)
            result.append([0,1,0])
            #print((tensor[0].shape))
        for item in path_B:
            img = image.load_img(item, target_size=(64,64))
            img = image.load_img(item, target_size=(64,64))
            img1 = np.asarray(img)
            lower_blue = np.array([3,50,50])
            upper_blue = np.array([33,255,255])
            hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            tensor.append(mask)
            result.append([0,0,1])
            #print((tensor[0].shape))
        np.save('result_try3',result)
        np.save('tensor_data_modified_try3',tensor)
        
    return tensor,result

def model_creation():
        model = Sequential()
        
        model.add(Conv2D(input_shape = [64,64,1],filters = 32,kernel_size = [5,5],padding = 'same',activation = 'relu'))
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
        model = load_model('hand_guesture_try3.h5')
    except:
        history = model.fit(image_matrices,result_values,epochs = 20,batch_size = 512,verbose = 1,validation_split = 0.1,shuffle = True)
        model.save('hand_guesture_try3.h5')
    return model
            

if __name__ == "__main__":
    path_A,path_B,path_C = preprocessing()
    print(len(path_A),len(path_B),len(path_C))
    image_matrices,result_values= image_to_matrix(path_A,path_B,path_C)
    print(image_matrices.shape,result_values.shape)
    model = model_creation()
    print(model.summary())
    model_trained = model_call(model,image_matrices,result_values)
    #model.fit(image_matrices,result_values,epochs = 20,batch_size = 512,verbose = 1,validation_split = 0.3)
    matric = []
    item = '11.jpg'
    img = image.load_img(item, target_size=(64,64))
    img1 = np.asarray(img)
    lower_blue = np.array([0,25,0])
    upper_blue = np.array([35,180,0])
    # Threshold the HSV image to get only blue colors
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    print(hsv.shape,'hsv_shape')
    mask = np.asarray(cv2.inRange(hsv, lower_blue, upper_blue))
    
    plt.imshow(mask, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    print(mask.shape,'mask_shape')
    mask = np.reshape(mask,(64,64,1))
    matric.append(mask)
    matric = np.asarray(matric)
    X = model_trained.predict(matric)
    print((X))
    for i in range (2):
        if int(X[0][i]) == 1:
            print('belongs to class',i)