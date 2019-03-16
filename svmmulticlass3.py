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


# In[9]:


from random import shuffle
from svmutil import *
percentvali = 2000
xarrtrainall = []
yarrtrainall = []
xarrvalidationall = []
yarrvalidationall = []
randomarr = list(range(0,numtrainall))
shuffle(randomarr)
randomarr = randomarr[:percentvali]
for x in randomarr:
    xarrvalidationall.append(xarrall[x])
    yarrvalidationall.append(yarrall[x])
    
for x in range(numtrainall):
    if(x not in randomarr):
        xarrtrainall.append(xarrall[x])
        yarrtrainall.append(yarrall[x])
carr = [0.00001, 0.001, 1, 5, 10]
print("xtrainalllength " + str(len(xarrtrainall)))
print("ytrainalllength " + str(len(yarrtrainall)))
print("xvalialllength " + str(len(xarrvalidationall)))
print("yvalinalllength " + str(len(yarrvalidationall)))
prob = svm_problem(yarrtrainall, xarrtrainall)
for x in range(5):
    print("lets see")
    print(carr[x])
    # param = svm_parameter('-s 0 -c 1.0 -t 0')
    paramstr = '-s 0 -c '+str(carr[x])+' -t 2 -g 0.05'
    print(paramstr)
    param = svm_parameter(paramstr)

    m=svm_train(prob, param)

    # m.predict(xarrtest[0])
    predy1 = svm_predict(yarrvalidationall, xarrvalidationall, m)
    predy2 = svm_predict(yarrtestall,xarrtestall,m)


