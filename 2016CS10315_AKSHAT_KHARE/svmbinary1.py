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
# yarr = np.array(yarr)
# print(xarr[0])
# print(xarr.shape)
# print(yarr.shape)
yarrsq=np.zeros((numtrain,numtrain))
for x in range(numtrain):
    yarrsq[x][x]=yarr[x]
P= np.matmul(np.matmul(yarrsq,np.matmul(xarr, np.transpose(xarr))),yarrsq)
# P=np.zeros((numtrain,numtrain))
# for x in range(numtrain):
#     for y in range(numtrain):
#         P[x][y] = (yarr[x])*(yarr[y])*(np.dot(xarr[x],xarr[y]))
# print("p shape is ")
# print(P.shape)
P=matrix(P)
if(debugcvx): print(P)
q=np.zeros((numtrain,1))
for x in range(numtrain):
    q[x][0] =-1.0
q=matrix(q)
if(debugcvx): print(q)
G=np.zeros((2*numtrain,numtrain))
for x in range(numtrain):
    G[x][x]=-1.0
    G[numtrain+x][x]=1.0
G=matrix(G)
if(debugcvx): print(G)
h=np.zeros((2*numtrain,1))
for x in range(numtrain):
    h[x][0]=0.0
    h[numtrain+x][0]=C
h=matrix(h)
if(debugcvx): print(h)
A=np.zeros((1,numtrain))
for x in range(numtrain):
    A[0][x] = yarr[x]
A=matrix(A)
if(debugcvx): print(A)
# print("A is")
# print(A)
b=matrix(0.0)
if(debugcvx): print(b)
    

starttime = time.time()
sol= solvers.qp(P,q,G,h,A,b)
# print(sol['x'])
alpha=np.array(sol['x'])
xarrsupport=[]
if(debugcvx): print(alpha)
warr=np.zeros((1,28*28))
for x in range(numtrain):
    # print(xarr[x])
    if(alpha[x][0]>pow(10,-4)):
        xarrsupport.append(xarr[x])
    warr = warr + (alpha[x][0])*(yarr[x])*(xarr[x])
    # print(warr)

tempbmax=float('-inf')
tempbmin=float('inf')
for x in range(numtrain):
#     print("warr and xarr shape are")
#     print(warr.shape)
#     print(xarr[x].shape)
    temp = float(np.dot(warr,xarr[x]))
    # print("tempdot is")
    # print(temp)
    if(yarr[x]==-1):
        if(temp>tempbmax):
            tempbmax=temp
    if(yarr[x]==1):
        if(temp<tempbmin):
            tempbmin=temp
bvalue = -(tempbmax+tempbmin)*0.5
endtime=time.time()
print("Training time is")
print(endtime-starttime)
writefile= open("svmbinarysupportvector.txt","w+")
writefile.write(str(xarrsupport))
# print("support vectors are")
# xarrsupport=np.array(xarrsupport)
# print(xarrsupport)
print("number of support vector vs total vectors")
print(len(xarrsupport))
print(numtrain)
writefile= open("svmbinaryw.txt","w+")
writefile.write(str(warr))
# warr=np.transpose(warr)
# print(warr.shape)
print("b is")
print(bvalue)
def findy(xtest):
    temp=0.0
    for x in range(numtrain):
        if(alpha[x][0]==0):
            continue
        temp+= (alpha[x][0])*(yarr[x])*(float(np.dot(xtest,xarr[x])))
    temp+=bvalue
    if(temp>=0):
        return 1.0
    else:
        return -1.0
preddigit=[]
read_file = open(sys.argv[2], "r")
count=0
numtest=0
yarrtest=[]
xarrtest=[]
correct=0
wrong=0
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
        temppredy = findy(np.array(tempxi))
        # print("pred is "+str(temppredy))
        # print("real is "+str(tempyi))
        if(temppredy==tempyi):
            correct+=1
        else:
            # print(str(wrong)+" wrong in "+str(numtest)+" steps")
            wrong+=1
        # if(numtest>2):
        #     break
xarrtest=np.array(xarrtest)
# yarrtest=np.array(yarrtest).reshape(numtest,1)
# yarrnp= np.array(yarr).reshape(numtrain,1)
print("Accuracy is")
print((1.0*correct)/(correct+wrong))
