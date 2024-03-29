import json
import numpy as np
import nltk
import math
import random
from nltk.tokenize import TweetTokenizer
read_file = open("ass2data/ass2_data/train.json", "r")
stararr=[]
freqarr=[]
# freqarr is the storage type for frequencies of words
# textarr=[]
worddict={}
# breaker=1
# worddict gives the direct index in freqarr corresponding to token
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
loopcount=1
for line in read_file:
    print("starting loop "+str(loopcount))
    data = json.loads(line)
    tempstars = int(data["stars"])
    temptext = (data["text"]).lower()
    # print(nltk.word_tokenize(temptext))
    # print(tknzr.tokenize(temptext))
    stararr.append(tempstars)
    # textarr.append(temptext)
    tokenizedarr = tknzr.tokenize(temptext)
    for x in range(len(tokenizedarr)):
        if (tokenizedarr[x]) in worddict:
            targetindex = worddict[tokenizedarr[x]]+tempstars-1
            freqarr[targetindex]=freqarr[targetindex]+1
        else:
            temparr = [0,0,0,0,0]
            temparr[tempstars-1]=1
            tempindex = len(freqarr)
            freqarr.extend(temparr)
            # print(freqarr)
            worddict[tokenizedarr[x]] = tempindex
    print("ending loop "+str(loopcount))
    loopcount+=1

    # if(loopcount>2000):
    #     print("custombreak")
    #     break


# print(freqarr)
print(len(freqarr))
print(len(worddict))
# print(worddict)
# for x,y in worddict.items():
#     print(x)
#     print(freqarr[y:y+5])
totalfreq=[0,0,0,0,0]
for x in range(int(len(freqarr)/5)):
    for y in range(5):
        totalfreq[y]=totalfreq[y] + freqarr[5*x+y]
print(totalfreq)

freqRev = [0,0,0,0,0]
for rev in stararr:
    freqRev[rev-1] = freqRev[rev-1] +1
freqarrlen=len(freqarr)


numtokens= len(worddict)
def predtext(tokenarr,trystar):
    tempans=0.0
    for x in tokenarr:
        if x in worddict:
            targetindex = worddict[x]+trystar-1
            tempans += math.log(freqarr[targetindex]+1) #+1 for laplace
        # print("line 60")
        # print(totalfreq[trystar-1])
        tempans -= math.log(totalfreq[trystar-1]+numtokens)
    tempans+=math.log(freqRev[trystar-1])
    return tempans

predstars=[]
predstarsrandom=[]
# predstarsmost=[]
realstars=[]
read_filetest = open("ass2data/ass2_data/ass2_data/test.json", "r")
loopcount=1

def starsmost():
    tempindex=0
    temp=freqRev[0]
    for x in range(5):
        if(freqRev[x]>temp):
            temp = freqRev[x]
            tempindex=x
    return tempindex

predstarsmostvalue=starsmost()    

for line in read_filetest:
    print("started test loop "+str(loopcount))
    data= json.loads(line)
    realstars.append(int(data["stars"]))
    temptext = (data["text"]).lower()
    tokenizedarr = tknzr.tokenize(temptext)
    temppredarr=[]

    for x in range(5):
        temppredindi=predtext(tokenizedarr,x+1)
        temppredarr.append(temppredindi)
    tempmax=temppredarr[0]
    tempmaxindex=0
    for x in range(5):
        if(temppredarr[x]>tempmax):
            tempmaxindex=x
            tempmax = temppredarr[x]
    print("real is "+str(data["stars"]))
    print("predicted is "+str(tempmaxindex+1))
    predstars.append(tempmaxindex+1)
    predstarsrandom.append(int(random.randint(0, 5)))
    # predstarsmost.append(predstarsmostvalue+1)
    print("ending test loop "+str(loopcount))
    loopcount+=1
    # if(loopcount>20):
    #     print("custom end to test")
    #     break

# calculate accuracy
correct=0
wrong=0
correctrandom=0
wrongrandom=0
correctmost=0
wrongmost=0
for x in range(len(predstars)):
    if(predstars[x]==realstars[x]):
        correct+=1
    else:
        wrong+=1
    if(predstarsrandom[x]==realstars[x]):
        correctrandom+=1
    else:
        wrongrandom+=1
    if((predstarsmostvalue+1)==realstars[x]):
        correctmost+=1
    else:
        wrongmost+=1
print("Naive Accuracy is")
print((1.0*correct)/(correct+wrong))
print("Random Accuracy is")
print((1.0*correctrandom)/(correctrandom+wrongrandom))
print("Most Accuracy is")
print((1.0*correctmost)/(correctmost+wrongmost))
print("Most one is "+str(predstarsmostvalue+1))

#Confusion matrix

confusionM=np.zeros((5,5))
for x in range(len(predstars)):
    i=predstars[x]-1
    j=realstars[x]-1
    # print("i is "+str(i))
    # print("j is "+str(j))
    confusionM[i][j] = confusionM[i][j]+1
    # print(confusionM)
print(confusionM)




