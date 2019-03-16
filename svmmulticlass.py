#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
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
read_file = open("ass2data/mnist/train.csv", "r")
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


# In[2]:



def findPGaussian(xarrarg,numtrainarg,yarrsqarg):
#     if(debuggausssion): print("xarrarg is")
#     if(debuggausssion): print(xarrarg)
    tempxirow = np.sum(np.multiply(xarrarg,xarrarg),axis=1).reshape((numtrainarg,1))
#     if(debuggausssion): print("tempxirow is")
#     if(debuggausssion): print(tempxirow)
    tempwhole = tempxirow + np.transpose(tempxirow) 
#     if(debuggausssion): print("tempwhole before dot add is")
#     if(debuggausssion): print(tempwhole)
#     if(debuggausssion): print("dot add is")
#     if(debuggausssion): print(((-2)*(np.matmul(xarrarg,np.transpose(xarrarg)))))
    tempwhole = tempwhole + ((-2)*(np.matmul(xarrarg,np.transpose(xarrarg))))
#     if(debuggausssion): print("tempwhole after dot add is")
#     if(debuggausssion): print(tempwhole)
    tempwhole = (-gammagaussian)*(tempwhole)
#     if(debuggausssion): print("tempwhole for exp is")
#     if(debuggausssion): print(tempwhole)
    tempwhole=np.exp(tempwhole)
    tempwhole = np.matmul(np.matmul(yarrsqarg,tempwhole),yarrsqarg)
#     if(debuggausssion): print(numtrainarg)
#     if(debuggausssion): print(tempwhole.shape)
    return tempwhole
starttime= time.time()
for x in range(10):
    for y in range(10):
        if(x>=y):
            continue
        print("x is " +str(x) + " y is "+ str(y))
        classindex = classtoindex[10*x+y]
        numtrainthis =numtrainclass[classindex]
        yarrsq=np.diag(yarrclass[classindex])
        P=matrix(findPGaussian(xarrclass[classindex], numtrainthis, yarrsq))
        q=np.zeros((numtrainthis,1))
        for z in range(numtrainthis):
            q[z][0] =-1.0
        q=matrix(q)
        G=np.zeros((2*numtrainthis,numtrainthis))
        for z in range(numtrainthis):
            G[z][z]=-1.0
            G[numtrainthis+z][z]=1.0
        G=matrix(G)
        h=np.zeros((2*numtrainthis,1))
        for z in range(numtrainthis):
            h[z][0]=0.0
            h[numtrainthis+z][0]=C
        h=matrix(h)
        A=np.zeros((1,numtrainthis))
        for z in range(numtrainthis):
            A[0][z] = yarrclass[classindex][z]
        A=matrix(A)
        b=matrix(0.0)
        sol= solvers.qp(P,q,G,h,A,b)
        alpha=np.array(sol['x'])
        alphaclass.append(alpha)
        print(alpha.shape)
        
print(alphaclass)


# In[3]:


debugygaussian = False
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

def findbgauss(alphaarg, numtrainarg, xarrargg, yarrargg):
    xarrsupport=[]
    yarrsupport=[]
    alphasupport=[]
    numsupport=0
    for x in range(numtrainarg):
        if(alphaarg[x][0]>pow(10,-4)):
            xarrsupport.append(xarrargg[x])
            yarrsupport.append(yarrargg[x])
            alphasupport.append(alphaarg[x][0])
            numsupport+=1
#             if(numsupport>1):
#                 break
    print("numsupport vector is "+str(numsupport)+" numtrain is "+str(numtrainarg))
#     if(numsupport==0):
#         print("forcefully returning 0 as b")
#         return 0
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

bclass=[]
for x in range(10):
    for y in range(10):
        if(x>=y):
            continue
        bclass.append(0.0)
for x in range(10):
    for y in range(10):
        if(x>=y):
            continue
        classindex = classtoindex[10*x+y]
        numtrainthis =numtrainclass[classindex]
        print("x is "+str(x)+" y is "+str(y))
        print(alphaclass[classindex])
        tempb = findbgauss(alphaclass[classindex], numtrainclass[classindex], xarrclass[classindex], yarrclass[classindex])
        bclass[classindex]=tempb
endtime=time.time()
print("Training time is")
print(endtime-starttime)
print(bclass)


# In[4]:


xarrtestall=[]
yarrtestall=[]
numtestall=0
ypredclassgauss=[]
ypredfin=[]
read_file = open("ass2data/mnist/test.csv", "r")
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


# In[5]:


for x in range(10):
    for y in range(10):
        if(x>=y):
            continue
        print("x is "+str(x)+ " y is "+str(y))
        classindex = classtoindex[10*x+y]
#         print(xarrclass[classindex])
#         print()
        ypredgaussian = findygaussian(xarrclass[classindex],np.array(xarrtestall),numtrainclass[classindex],numtestall,yarrclass[classindex],alphaclass[classindex])
        ypredgaussian = ypredgaussian + bclass[classindex]
        ypredclassgauss.append(ypredgaussian)

for testiter in range(numtestall):
    numwins = []
    scorearr = []
    for x in range(10):
        numwins.append(0)
        scorearr.append(0.0)
    for x in range(10):
        for y in range(10):
            if(x>=y):
                continue
            classindex = classtoindex[10*x+y]
            if(ypredclassgauss[classindex][testiter]>=0):
                numwins[y]= numwins[y] +1
                scorearr[y] = scorearr[y] +ypredclassgauss[classindex][testiter]
            else:
                numwins[x] = numwins[x] +1
                scorearr[x] = scorearr[x] -ypredclassgauss[classindex][testiter]
    
    maxwinindex=0
    maxwin=numwins[0]
    scoreofmaxwin = scorearr[0]
    for x in range(1,10):
        if(numwins[x]==maxwin):
            if(scorearr[x]>scoreofmaxwin):
                maxwinindex=x
                maxwin=numwins[x]
                scoreofmaxwin = scorearr[x]
        elif(numwins[x]>maxwin):
            maxwinindex=x
            maxwin=numwins[x]
            scoreofmaxwin = scorearr[x]
    ypredfin.append(maxwinindex)
print(ypredfin)


# In[6]:


correct=0
wrong=0
for testiter in range(numtestall):
    print(str(ypredfin[testiter])+ " " + str(yarrtestall[testiter]))
    if(ypredfin[testiter]==yarrtestall[testiter]):
        correct+=1
    else:
        wrong+=1
print("Accuracy is")
print((1.0*correct)/(correct+wrong))


# In[7]:


confusionM=np.zeros((10,10))
# print(yarrtestall)

for x in range(numtestall):
    i=int(ypredfin[x])
    j=int(yarrtestall[x])
#     print("i is "+str(i))
#     print("j is "+str(j))
    confusionM[i][j] = confusionM[i][j]+1
    # print(confusionM)
print(confusionM.astype(int))


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

