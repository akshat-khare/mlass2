import json
import numpy as np
import nltk
import math
import sys
from nltk.tokenize import TweetTokenizer
read_file = open(sys.argv[1], "r")
stararr=[]
freqarr=[]
debug=False
# freqarr is the storage type for frequencies of words
# textarr=[]
worddict={}
# worddict gives the direct index in freqarr corresponding to token
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
loopcount=1
for line in read_file:
    if(debug): print("starting loop "+str(loopcount))
    data = json.loads(line)
    tempstars = int(data["stars"])
    temptext = data["text"]
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
    if(debug): print("ending loop "+str(loopcount))
    loopcount+=1

    if(loopcount>2000):
        print("custombreak")
        break


# print(freqarr)
if(debug): print(len(freqarr))
if(debug): print(len(worddict))
# print(worddict)
# for x,y in worddict.items():
#     print(x)
#     print(freqarr[y:y+5])
totalfreq=[0,0,0,0,0]
for x in range(int(len(freqarr)/5)):
    for y in range(5):
        totalfreq[y]=totalfreq[y] + freqarr[5*x+y]
if(debug): print(totalfreq)

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
realstars=[]
read_filetest = open(sys.argv[2], "r")
loopcount=1
for line in read_filetest:
    if(debug): print("started test loop "+str(loopcount))
    data= json.loads(line)
    realstars.append(int(data["stars"]))
    temptext = data["text"]
    tokenizedarr = tknzr.tokenize(temptext)
    temppredarr=[]

    for x in range(5):
        temppredindi=predtext(tokenizedarr,x+1)
        temppredarr.append(temppredindi)
    tempmax=temppredarr[0];
    tempmaxindex=0
    for x in range(5):
        if(temppredarr[x]>tempmax):
            tempmaxindex=x
            tempmax = temppredarr[x]
    if(debug): print("real is "+str(data["stars"]))
    if(debug): print("predicted is "+str(tempmaxindex+1))
    predstars.append(tempmaxindex+1)
    if(debug): print("ending test loop "+str(loopcount))
    loopcount+=1
    if(loopcount>20):
        print("custom end to test")
        break

# calculate accuracy
correct=0
wrong=0
for x in range(len(predstars)):
    if(predstars[x]==realstars[x]):
        correct+=1
    else:
        wrong+=1
print("Accuracy is")
print((1.0*correct)/(correct+wrong))




