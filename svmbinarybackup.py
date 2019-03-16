import numpy as np
from cvxopt import matrix, solvers
considerata=3
C=1.0
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
#         if(numtrain>20):
#             break
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
print("p shape is ")
print(P.shape)
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
            print(str(wrong)+" wrong in "+str(numtest)+" steps")
            wrong+=1
#         if(numtest>2):
#             break
xarrtest=np.array(xarrtest)
# yarrtest=np.array(yarrtest).reshape(numtest,1)
# yarrnp= np.array(yarr).reshape(numtrain,1)
print("Accuracy is")
print((1.0*correct)/(correct+wrong))





# p shape is 
# (4000, 4000)
#      pcost       dcost       gap    pres   dres
#  0: -3.4336e+02 -7.6347e+03  4e+04  3e+00  7e-13
#  1: -2.0201e+02 -4.0904e+03  8e+03  4e-01  5e-13
#  2: -9.5671e+01 -1.4111e+03  2e+03  1e-01  3e-13
#  3: -4.6005e+01 -6.1590e+02  1e+03  4e-02  2e-13
#  4: -2.4389e+01 -3.6744e+02  6e+02  2e-02  1e-13
#  5: -1.3241e+01 -2.7558e+02  4e+02  1e-02  8e-14
#  6: -3.5771e+00 -1.8538e+02  3e+02  6e-03  6e-14
#  7: -1.8094e+00 -5.3678e+01  7e+01  1e-03  4e-14
#  8: -2.5254e+00 -3.1071e+01  4e+01  5e-04  3e-14
#  9: -9.8362e-01 -2.3127e+01  2e+01  5e-05  3e-14
# 10: -3.8740e+00 -1.3199e+01  1e+01  1e-05  3e-14
# 11: -5.2616e+00 -9.6657e+00  4e+00  2e-06  4e-14
# 12: -5.7900e+00 -8.3199e+00  3e+00  1e-15  4e-14
# 13: -6.4810e+00 -7.2514e+00  8e-01  4e-15  4e-14
# 14: -6.6492e+00 -6.9700e+00  3e-01  4e-15  4e-14
# 15: -6.7800e+00 -6.8183e+00  4e-02  7e-16  4e-14
# 16: -6.7974e+00 -6.7983e+00  9e-04  2e-16  4e-14
# 17: -6.7978e+00 -6.7978e+00  1e-05  2e-15  4e-14
# 18: -6.7978e+00 -6.7978e+00  1e-07  2e-16  4e-14
# Optimal solution found.
# 0 wrong in 451 steps
# 1 wrong in 582 steps
# 2 wrong in 733 steps
# 3 wrong in 924 steps
# 4 wrong in 1016 steps
# 5 wrong in 1028 steps
# 6 wrong in 1146 steps
# 7 wrong in 1175 steps
# 8 wrong in 1493 steps
# 9 wrong in 1567 steps
# Accuracy is
# 0.9949799196787149


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
sol= solvers.qp(P,q,G,h,A,b)
# print(sol['x'])
alpha=np.array(sol['x'])
if(debugcvx): print(alpha)
warr=np.zeros((1,28*28))
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




#      pcost       dcost       gap    pres   dres
#  0: -1.3972e+02 -5.5951e+03  2e+04  2e+00  2e-15
#  1: -7.0083e+01 -2.6315e+03  4e+03  2e-01  1e-15
#  2: -1.0457e+02 -4.7107e+02  4e+02  1e-02  3e-15
#  3: -1.4846e+02 -2.3899e+02  9e+01  2e-03  2e-15
#  4: -1.6174e+02 -1.8655e+02  3e+01  4e-04  1e-15
#  5: -1.6639e+02 -1.7384e+02  8e+00  7e-05  9e-16
#  6: -1.6824e+02 -1.6971e+02  1e+00  3e-14  9e-16
#  7: -1.6865e+02 -1.6877e+02  1e-01  6e-15  9e-16
#  8: -1.6869e+02 -1.6870e+02  4e-03  2e-14  9e-16
#  9: -1.6870e+02 -1.6870e+02  8e-05  6e-15  9e-16
# Optimal solution found.
# numsupport vector is 1365 numtrain is 4000
# b is 
# -0.1948271424107001
# Accuracy is
# 0.998995983935743


from svmutil import *
# svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

prob = svm_problem(yarr, xarr)

# param = svm_parameter('-s 0 -c 1.0 -t 0')
param = svm_parameter('-s 0 -c 1.0 -t 2 -g 0.05')

m=svm_train(prob, param)

# m.predict(xarrtest[0])
correct=0
wrong=0
for x in range(numtest):
    predy = m.predict(xarrtest[x])
    if(predy==yarrtest[x]):
        correct+=1
    else:
        wrong+=1

print("Accuracy is")
print((1.0*correct)/(correct+wrong))




# Accuracy is
# 0.998995983935743