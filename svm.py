import numpy as np
from cvxopt import matrix, solvers
considerata=3
read_file = open("ass2data/mnist/train.csv", "r")
lenlinarr=28*28+1
xarr=[]
yarr=[]
numtrain=0
debugcvx=False
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
P=matrix(P)
if(debugcvx): print(P)
q=np.zeros((numtrain,1))
for x in range(numtrain):
    q[x][0] =-1.0
q=matrix(q)
if(debugcvx): print(q)
G=np.zeros((numtrain,numtrain))
for x in range(numtrain):
    G[x][x]=-1.0
G=matrix(G)
if(debugcvx): print(G)
h=np.zeros((numtrain,1))
for x in range(numtrain):
    h[x][0]=0.0
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
sol= solvers.qp(P,q,G,h,A,b)
# print(sol['x'])
alpha=np.array(sol['x'])
if(debugcvx): print(alpha)
warr=np.zeros((1,28*28))
for x in range(numtrain):
    # print(xarr[x])
    warr = warr + (alpha[x][0])*(yarr[x])*(xarr[x])
    # print(warr)
# warr=np.transpose(warr)
# print(warr)
tempbmax=float('-inf')
tempbmin=float('inf')
for x in range(numtrain):
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
# print("b is")
# print(bvalue)
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
read_file = open("ass2data/mnist/test.csv", "r")
count=0
numtest=0
# yarrtest=[]
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
        # xarr.append(tempxi)
        if(tempyi==considerata+1):
            tempyi=1.0
        else:
            tempyi=-1.0
        # yarrtest.append(tempyi)
        # print(xarr)

        numtest+=1
        temppredy = findy(np.array(tempxi))
        # print("pred is "+str(temppredy))
        # print("real is "+str(tempyi))
        if(temppredy==tempyi):
            correct+=1
        else:
            print(str(wrong)+" wrong in "+str(numtest)+" steps")
            wrong+=1
        # if(numtest>2):
        #     break
print("Accuracy is")
print((1.0*correct)/(correct+wrong))
def findPGaussian():
    tempxirow = np.sum(np.multiply(xarr,xarr),axis=1).reshape((numtrain,1))
    tempwhole = tempxirow + np.transpose(tempxirow) + ((-2)*(np.dot(xarr,np.transpose(xarr))))
    tempwhole = np.matmul(np.matmul(yarrsq,tempwhole),yarrsq)
    return tempwhole
P=matrix(findPGaussian())

