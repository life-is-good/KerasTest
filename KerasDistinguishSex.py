# -*- coding: utf-8 -*-  
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import csv
import codecs
from time import sleep
batch_size = 128
nb_classes = 2
nb_epoch = 10
def load_data():
   x_train=[]
   Y_train=[]
   x_test=[]
   Y_test=[]
   
   f = open("distinguishsexdata/train.txt","r")
   i = 0
   for line in f.readlines():
       line = line.split(",")
       if i>0:
         if line[1] == "男":
             Y_train.append(0)
         else: 
             Y_train.append(1)
         del line[0]
         del line[0]
         del line[0]
         x_train.append(line)
       i += 1
   x1=np.array(x_train)
   y1=np.array(Y_train)
   f.close()
   
   f = open("distinguishsexdata/test.txt","r")
   i = 0
   for line in f.readlines():
       line = line.split(",")
       if i>0:
         if line[1] == "男":
             Y_test.append(0)
         else:
             Y_test.append(1)
         del line[0]
         del line[0]
         del line[0]
         x_test.append(line)
       i += 1
   x2=np.array(x_test)
   y2=np.array(Y_test)
   f.close()
   return (x1, y1),(x2, y2)
#    return (x1, y1), (x2, y2),Y_test
   
# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test),t = load_data()
(X_train, y_train), (X_test, y_test) = load_data()
X_train = X_train.reshape(1858, 26)
X_test = X_test.reshape(200, 26)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#分成3层，中间隐层有512个节点
model = Sequential()
model.add(Dense(32, input_shape=(26,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=32))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=2))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
history = model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test))

# count = 0
# result = []
# result = model.predict_classes(X_test,batch_size=200,verbose=1)
# for i in range(len(result)):
#     if result[i] == t[i]:
#         count +=1
# print ("same count:",count)

score = model.evaluate(X_test, Y_test, verbose=1)
#print('Test score:', score[0])
print('Test accuracy:', score[1])
print "end"
