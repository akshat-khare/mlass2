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
for x in range(numtrain):
    # print(xarr[x])
    warr = warr + (alpha[x][0])*(yarr[x])*(xarr[x])
    # print(warr)
# warr=np.transpose(warr)
# print(warr)
supportvectorindex=0
for x in range(numtrain):
    if(alpha[x][0]>0):
        supportvectorindex=x
        break
# bvalue = yarr[supportvectorindex]-float(np.dot(warr,xarr[supportvectorindex]))
bvalue=0
# def kernel(xi,xj):
#     temp = np.matmul(np.transpose(xi-xj))
debugygaussian=False
def findygaussian():
    temp1 = np.transpose(np.sum(np.multiply(xarr,xarr),axis=1).reshape((numtrain,1)))
    
    if(debugygaussian): print("temp1 is")
    if(debugygaussian): print(temp1)
        
    temp2 = np.sum(np.multiply(xarrtest,xarrtest),axis=1).reshape((numtest,1))
    
    if(debugygaussian): print("temp2 is")
    if(debugygaussian): print(temp2)
    
    temp3 = (-2)*(np.dot(xarrtest,np.transpose(xarr)))
    
    if(debugygaussian): print("temp3 is")
    if(debugygaussian): print(temp3)
    
    temp4 = temp1+temp2+temp3
    
    if(debugygaussian): print("temp4 is")
    if(debugygaussian): print(temp4)
    
    temp4 = (-gammagaussian)*(temp4)
    temp4 = np.exp(temp4)
    
    if(debugygaussian): print("alpha is")
    if(debugygaussian): print(alpha)
    if(debugygaussian): print("yarr is")
    if(debugygaussian): print(yarr)
    
    temp5 = np.multiply(alpha,np.array(yarr).reshape((numtrain,1)))
    
    if(debugygaussian): print("temp5 is =========")
    if(debugygaussian): print(temp5)
#     temp5 = np.diag(np.transpose(temp5)[0])
    
    temp4 = np.matmul(temp4,temp5)
    
    if(debugygaussian): print("prefinal temp4 is")
    if(debugygaussian): print(temp4)
    
    temp4 = temp4 + bvalue
    
    return temp4
ypredgaussian = findygaussian()
debugygaussian2=True
if(debugygaussian2): print("ypredgaussian is")
if(debugygaussian2): print(ypredgaussian)
correct=0
wrong=0
for x in range(numtest):
    if((ypredgaussian[x][0])*(yarrtest[x]) >=0):
        print("found right for "+str(yarrtest[x]) + "val was "+str(ypredgaussian[x][0]))
        correct+=1
    else:
        print("found wrong for "+str(yarrtest[x]) + "val was "+str(ypredgaussian[x][0]))
        wrong+=1
print("Accuracy is")
print((1.0*correct)/(correct+wrong))