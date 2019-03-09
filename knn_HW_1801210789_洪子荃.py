#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:38:43 2018

@author: hzchuan
"""
#HW2_1_KNN
import pandas as pd
import numpy as np
import operator
import collections
Letter_Recog = pd.read_csv('letter-recognition.csv', header=None)
#16個Attributes
X = Letter_Recog.iloc[:,1:] #DATA X
Y = Letter_Recog[0] #DATA Y
#隨機切資料
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(Ｘ,Ｙ,test_size=0.2)
#分類機訓練
def classify0(test, Xtrain, ytrain, k):	
	dist = np.sum((test - Xtrain)**2, axis=1)**0.5 # 
	k_labels = [ytrain[i] for i in dist.sort_values().index[0:k]] # k個最近的標籤
	# 出現次數最多的標籤就為最後類別
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label
#if __name__ == '__main__': 
p = []
for a in Xtest.index[0:]:     
    test = Xtest.loc[a,:]
    y_hat = classify0(test, Xtrain, ytrain, 4)
    #print(y_hat) #預測結果
    p.append(y_hat)

#測試正確率   
ytest = ytest.tolist()
out = []
for i in range(len(p)):  
    out.append(operator.eq(p[i],ytest[i]))
print(str('正確率:'),np.sum(out)/len(p))



