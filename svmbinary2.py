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
        # if(numtrain>200):
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
        # if(numtest>20):
        #     break
xarrtest=np.array(xarrtest)







gammagaussian=0.05
debuggausssion=False
def findPGaussian():
    if(debuggausssion): print("xarr is")
    if(debuggausssion): print(xarr)
    tempxirow = np.sum(np.multiply(xarr,xarr),axis=1).reshape((numtrain,1))
    if(debuggausssion): print("tempxirow is")
    if(debuggausssion): print(tempxirow)
    tempwhole = tempxirow + np.transpose(tempxirow) 
    if(debuggausssion): print("tempwhole before dot add is")
    if(debuggausssion): print(tempwhole)
    if(debuggausssion): print("dot add is")
    if(debuggausssion): print(((-2)*(np.matmul(xarr,np.transpose(xarr)))))
    tempwhole = tempwhole + ((-2)*(np.matmul(xarr,np.transpose(xarr))))
    if(debuggausssion): print("tempwhole after dot add is")
    if(debuggausssion): print(tempwhole)
    tempwhole = (-gammagaussian)*(tempwhole)
    if(debuggausssion): print("tempwhole for exp is")
    if(debuggausssion): print(tempwhole)
    tempwhole=np.exp(tempwhole)
    tempwhole = np.matmul(np.matmul(yarrsq,tempwhole),yarrsq)
    if(debuggausssion): print(numtrain)
    if(debuggausssion): print(tempwhole.shape)
    return tempwhole
P=matrix(findPGaussian())
# print(P)
starttime=time.time()
sol= solvers.qp(P,q,G,h,A,b)
# print(sol['x'])
alpha=np.array(sol['x'])
if(debugcvx): print(alpha)
# warr=np.zeros((1,28*28))
# for x in range(numtrain):
#     # print(xarr[x])
#     warr = warr + (alpha[x][0])*(yarr[x])*(xarr[x])
#     # print(warr)
# # warr=np.transpose(warr)
# # print(warr)
# supportvectorindex=0
# for x in range(numtrain):
#     if(alpha[x][0]>pow(10,-4)):
#         supportvectorindex=x
#         break
# # bvalue = yarr[supportvectorindex]-float(np.dot(warr,xarr[supportvectorindex]))
# bvalue=0
# def kernel(xi,xj):
#     temp = np.matmul(np.transpose(xi-xj))
debugygaussian=False
def findygaussian(xarrargg,xarrtestargg,numtrainarg,numtestarg,yarrargg,alphaargg):
    temp1 = np.transpose(np.sum(np.multiply(xarrargg,xarrargg),axis=1).reshape((numtrainarg,1)))
    
    if(debugygaussian): print("temp1 is")
    if(debugygaussian): print(temp1)
        
    temp2 = np.sum(np.multiply(xarrtestargg,xarrtestargg),axis=1).reshape((numtestarg,1))
    
    if(debugygaussian): print("temp2 is")
    if(debugygaussian): print(temp2)
    
    temp3 = (-2)*(np.dot(xarrtestargg,np.transpose(xarrargg)))
    
    if(debugygaussian): print("temp3 is")
    if(debugygaussian): print(temp3)
    
    temp4 = temp1+temp2+temp3
    
    if(debugygaussian): print("temp4 is")
    if(debugygaussian): print(temp4)
    
    temp4 = (-gammagaussian)*(temp4)
    temp4 = np.exp(temp4)
    
    if(debugygaussian): print("alphaargg is")
    if(debugygaussian): print(alphaargg)
    if(debugygaussian): print("yarrargg is")
    if(debugygaussian): print(yarrargg)
    
    temp5 = np.multiply(alphaargg,np.array(yarrargg).reshape((numtrainarg,1)))
    
    if(debugygaussian): print("temp5 is =========")
    if(debugygaussian): print(temp5)
#     temp5 = np.diag(np.transpose(temp5)[0])
    
    temp4 = np.matmul(temp4,temp5)
    
    if(debugygaussian): print("prefinal temp4 is")
    if(debugygaussian): print(temp4)
    
    temp4 = temp4
    
    return temp4

# Lets find b
write_file= open("svmbinarysupportgaussian.txt", "w+")

def findbgauss():
    xarrsupport=[]
    yarrsupport=[]
    alphasupport=[]
    numsupport=0
    for x in range(numtrain):
        if(alpha[x][0]>pow(10,-4)):
            xarrsupport.append(xarr[x])
            yarrsupport.append(yarr[x])
            alphasupport.append(alpha[x][0])
            numsupport+=1
#             if(numsupport>1):
#                 break
    print("numsupport vector is "+str(numsupport)+" numtrain is "+str(numtrain))
    write_file.write(str(xarrsupport))
    xarrsupport=np.array(xarrsupport)
    alphasupport=np.transpose(np.array(alphasupport).reshape((1,numsupport)))
    kernalans = findygaussian(xarrsupport,xarrsupport,numsupport,numsupport,yarrsupport,alphasupport)
    yarrsupportnp = np.transpose(np.array(yarrsupport).reshape(1,numsupport))
    allbs = yarrsupportnp-kernalans
    temp=0.0
    for x in range(numsupport):
        temp+=allbs[x][0]
    temp=temp/numsupport
    return temp
bvalue=findbgauss()
endtime= time.time()
print("Training time is")
print(endtime-starttime)
print("b is ")
print(bvalue)
ypredgaussian = findygaussian(xarr,xarrtest,numtrain,numtest,yarr,alpha) + bvalue
debugygaussian2=False
if(debugygaussian2): print("ypredgaussian is")
if(debugygaussian2): print(ypredgaussian)
correct=0
wrong=0
for x in range(numtest):
    if((ypredgaussian[x][0])*(yarrtest[x]) >=0):
#         print("found right for "+str(yarrtest[x]) + "val was "+str(ypredgaussian[x][0]))
        correct+=1
    else:
#         print("found wrong for "+str(yarrtest[x]) + "val was "+str(ypredgaussian[x][0]))
        wrong+=1
print("Accuracy is")
print((1.0*correct)/(correct+wrong))
