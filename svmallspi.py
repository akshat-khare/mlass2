import numpy as np
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
        yarrclass[classindex].append(tempyi)
        xarrclass[classindex].append(tempxi)
        numtrainclass[classindex] = numtrainclass[classindex] +1
    for x in range(0,tempyi):
        classindex= classtoindex[10*x+tempyi]
        yarrclass[classindex].append(tempyi)
        xarrclass[classindex].append(tempxi)
        numtrainclass[classindex] = numtrainclass[classindex] +1
    
    numtrainall+=1
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

for x in range(10):
    for y in range(10):
        if(x>=y):
            continue
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
        
print(alphaclass)
        