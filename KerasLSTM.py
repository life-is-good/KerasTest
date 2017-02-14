# -*- coding: utf-8 -*-
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
 
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

neg=pd.read_excel('comments/neg.xls',header=None,index=None)
pos=pd.read_excel('comments/pos.xls',header=None,index=None) #读取训练语料完毕
pos['mark']=1
neg['mark']=0 #给训练语料贴上标签
pn=pd.concat([pos,neg],ignore_index=True) #合并语料
neglen=len(neg)
poslen=len(pos) #计算语料数目
 
cw = lambda x: list(jieba.cut(x)) #定义分词函数
pn['words'] = pn[0].apply(cw)
 
comment = pd.read_excel('comments/1.xls') #读入评论内容
comment = comment[comment[u'短评'].notnull()] #仅读取非空评论
comment['words'] = comment[u'短评'].apply(cw) #评论分词 

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True) 
 
w = [] #将所有词语整合在一起
for i in d2v_train:
  w.extend(i)
 
dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w,d2v_train
dict['id']=list(range(1,len(dict)+1))
 
get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent) #速度太慢
 
maxlen = 50
 
print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
 
x = np.array(list(pn['sent']))[::2] #训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['mark']))
 
print('Build model...')
model = Sequential()
model.add(Embedding(len(dict)+1, 256))
model.add(LSTM(output_dim=32, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(input_dim = 32, output_dim = 1))
model.add(Activation('sigmoid'))
print ('Model bulid complete...')

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary", metrics=['accuracy'])
print ("Model compile complete ...")

model.fit(x, y, batch_size=16, nb_epoch=1,show_accuracy=True,validation_data=(xt, yt)) #训练时间为若干个小时

classes = model.predict_classes(xa)
for c in classes:
    print c

score = model.evaluate(xt, yt, verbose=1)
print ("Test accuracy:",score[1])
# acc = np_utils.accuracy(classes, yt)
# print('Test accuracy:', acc)

