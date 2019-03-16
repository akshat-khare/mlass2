import json
import numpy as np
import nltk
import math
import random
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import ngrams
import sys
stop = set(stopwords.words('english'))
# add punctuation to stopword
# print(stop)
# exit()
ps = PorterStemmer()
read_file = open(sys.argv[1], "r")
stararr=[]
freqarr=[]
debug=False
# freqarr is the storage type for frequencies of words
# textarr=[]
worddict={}
ngramval=2
# breaker=1
# worddict gives the direct index in freqarr corresponding to token
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
def tokenmaker(textdata):
    #assume it is lowercase
    tokenizedarr = tknzr.tokenize(temptext)
    # print(tokenizedarr)
    tokenizedarrstemmed=[]
    for x in tokenizedarr:
        tokenizedarrstemmed.append(ps.stem(x))
    tokenizedarrstemmedstop=[]
    for x in tokenizedarrstemmed:
        if x not in stop:
            tokenizedarrstemmedstop.append(x)
    tokenizedarrstemmedstopngramtemp = ngrams(tokenizedarrstemmedstop,ngramval)
    tokenizedarrstemmedstopngram=[]
    for grams in tokenizedarrstemmedstopngramtemp:
        tokenizedarrstemmedstopngram.append(' '.join(grams))
    # print(tokenizedarrstemmedstopngram)
    return tokenizedarrstemmedstopngram

loopcount=1
for line in read_file:
    if(debug): print("starting loop "+str(loopcount))
    data = json.loads(line)
    tempstars = int(data["stars"])
    temptext = (data["text"]).lower()
    # print(nltk.word_tokenize(temptext))
    # print(tknzr.tokenize(temptext))
    stararr.append(tempstars)
    # textarr.append(temptext)
    tokenizedarr = tokenmaker(temptext)
    # print(tokenizedarr)
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

    # if(loopcount>2000):
    #     print("custombreak")
    #     break


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
predstarsrandom=[]
# predstarsmost=[]
realstars=[]
read_filetest = open(sys.argv[2], "r")
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
    if(debug): print("started test loop "+str(loopcount))
    data= json.loads(line)
    realstars.append(int(data["stars"]))
    temptext = (data["text"]).lower()
    tokenizedarr = tokenmaker(temptext)
    # print(tokenizedarr)
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
    if(debug): print("real is "+str(data["stars"]))
    if(debug): print("predicted is "+str(tempmaxindex+1))
    predstars.append(tempmaxindex+1)
    predstarsrandom.append(int(random.randint(0, 5)))
    # predstarsmost.append(predstarsmostvalue+1)
    if(debug): print("ending test loop "+str(loopcount))
    loopcount+=1
    # if(loopcount>200):
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
print("Confusion Matrix is")
confusionM=np.zeros((5,5))
for x in range(len(predstars)):
    i=predstars[x]-1
    j=realstars[x]-1
    # print("i is "+str(i))
    # print("j is "+str(j))
    confusionM[i][j] = confusionM[i][j]+1
    # print(confusionM)
print(confusionM)

indif1 = 0.0
indifarr=[]
for x in range(5):
    temp=0.0
    for y in range(5):
        temp +=confusionM[x][y]
        temp +=confusionM[y][x]
    temp = (2*(confusionM[x][x])*1.0)/(temp)
    print("f1 for "+str(x+1))
    print(temp)
    indifarr.append(temp)
    indif1+=temp
indif1= indif1/5
print("F1 array is ")
print(indifarr)
print("Average F1 score is ")
print(indif1)




