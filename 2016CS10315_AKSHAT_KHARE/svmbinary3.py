#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
from cvxopt import matrix, solvers
import sys
considerata=5
C=1.0
read_file = open(sys.argv[1], "r")
lenlinarr=28*28+1
xarr=[]
yarr=[]
numtrain=0
debugcvx=False
trainingtime=0.0
for line in read_file:
    linearr=line.split(',')
    tempyi=int(linearr[lenlinarr-1])
    # print(tempyi)
    if(tempyi==considerata or tempyi==considerata+1):
        # print(tempyi)
        tempxi= []
        for x in range(28*28):
            tempxi.append(1.0*int(linearr[x])/255)
        # print(tempxi)
        xarr.append(tempxi)
        if(tempyi==considerata+1):
            tempyi=1.0
        else:
            tempyi=-1.0
        yarr.append(tempyi)
        # print(xarr)
        numtrain+=1
        # if(numtrain>20):
        #     break
xarr = np.array(xarr)

read_file = open(sys.argv[2], "r")
numtest=0
yarrtest=[]
xarrtest=[]
for line in read_file:
    linearr=line.split(',')
    tempyi=int(linearr[lenlinarr-1])
    # print(tempyi)
    if(tempyi==considerata or tempyi==considerata+1):
        # print(tempyi)
        tempxi= []
        for x in range(28*28):
            tempxi.append(1.0*int(linearr[x])/255)
        # print(tempxi)
        xarrtest.append(tempxi)
        if(tempyi==considerata+1):
            tempyi=1.0
        else:
            tempyi=-1.0
        yarrtest.append(tempyi)
        # print(xarr)

        numtest+=1
        # if(numtest>2):
        #     break
xarrtest=np.array(xarrtest)





from svmutil import *
# svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

prob = svm_problem(yarr, xarr)

param = svm_parameter('-s 0 -c 1.0 -t 0')
# param = svm_parameter('-s 0 -c 1.0 -t 2 -g 0.05')

starttime=time.time()
m=svm_train(prob, param)
endtime=time.time()
print("Traintime is")
print(endtime-starttime)
# print("nsv")
# print(len(m.nSV))
# print(type(m.nSV))

predy = svm_predict(yarrtest,xarrtest,m)

# param = svm_parameter('-s 0 -c 1.0 -t 0')
param = svm_parameter('-s 0 -c 1.0 -t 2 -g 0.05')

starttime=time.time()

m=svm_train(prob, param)

endtime=time.time()
print("Traintime is")
print(endtime-starttime)
# print("nsv")
# print(m.nSV)
# print(type(m.nSV))

predy = svm_predict(yarrtest,xarrtest,m)
