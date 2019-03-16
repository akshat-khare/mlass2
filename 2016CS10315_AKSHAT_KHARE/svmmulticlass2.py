#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import sys
from cvxopt import matrix, solvers
lenlinarr=28*28+1
gammagaussian=0.05
C=1.0
xarrall=[]
yarrall=[]
numtrainall=0
xarrclass=[]
yarrclass=[]
numtrainclass=[]
alphaclass=[]
classtoindex={}
numindex=0
for x in range(10):
    for y in range(10):
        if(x>=y):
            continue
        yarrclass.append([])
        xarrclass.append([])
        classtoindex[10*x+y]=numindex
        numtrainclass.append(0)
        numindex+=1
read_file = open(sys.argv[1], "r")
for line in read_file:
    linearr=line.split(',')
    tempyi=int(linearr[lenlinarr-1])
    # print(tempyi)
    tempxi= []
    for x in range(28*28):
        tempxi.append(1.0*int(linearr[x])/255)
    # print(tempxi)
    xarrall.append(tempxi)
    yarrall.append(tempyi)
    # print(xarr)
    for x in range(tempyi+1,10):
        classindex= classtoindex[10*tempyi+x]
        yarrclass[classindex].append(-1.0)
        xarrclass[classindex].append(tempxi)
        numtrainclass[classindex] = numtrainclass[classindex] +1
    for x in range(0,tempyi):
        classindex= classtoindex[10*x+tempyi]
        yarrclass[classindex].append(1.0)
        xarrclass[classindex].append(tempxi)
        numtrainclass[classindex] = numtrainclass[classindex] +1
    
    numtrainall+=1
    # if(numtrainall>200):
    #     break


# In[4]:


xarrtestall=[]
yarrtestall=[]
numtestall=0
ypredclassgauss=[]
ypredfin=[]
read_file = open(sys.argv[2], "r")
for line in read_file:
    linearr=line.split(',')
    tempyi=int(linearr[lenlinarr-1])
    # print(tempyi)
    tempxi= []
    for x in range(28*28):
        tempxi.append(1.0*int(linearr[x])/255)
    # print(tempxi)
    xarrtestall.append(tempxi)
    yarrtestall.append(tempyi)
    numtestall+=1
    # if(numtestall>20):
    #     break

# In[8]:


from svmutil import *
# svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

prob = svm_problem(yarrall, xarrall)

# param = svm_parameter('-s 0 -c 1.0 -t 0')
param = svm_parameter('-s 0 -c 1.0 -t 2 -g 0.05')

starttime=time.time()
m=svm_train(prob, param)
endtime=time.time()
print("Train time is")
print(endtime-starttime)
# m.predict(xarrtest[0])
predy = svm_predict(yarrtestall,xarrtestall,m)

